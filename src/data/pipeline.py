"""
Data Pipeline: Merge, clean, and feature-engineer the Olist dataset
into a single training-ready DataFrame.

Usage:
    python -m src.data.pipeline
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ORDER_ITEMS_CSV, ORDERS_CSV, PRODUCTS_CSV, CUSTOMERS_CSV,
    REVIEWS_CSV, SELLERS_CSV, CATEGORY_TRANSLATION_CSV,
    FEATURES_PARQUET, LABEL_ENCODERS_PATH,
)
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Step 1: Load Raw Data ────────────────────────────────────

def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load all required raw CSV files into DataFrames."""
    logger.info("Loading raw CSV files...")
    tables = {
        "order_items": pd.read_csv(ORDER_ITEMS_CSV),
        "orders": pd.read_csv(ORDERS_CSV, parse_dates=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]),
        "products": pd.read_csv(PRODUCTS_CSV),
        "customers": pd.read_csv(CUSTOMERS_CSV),
        "reviews": pd.read_csv(REVIEWS_CSV),
        "sellers": pd.read_csv(SELLERS_CSV),
        "category_translation": pd.read_csv(CATEGORY_TRANSLATION_CSV),
    }
    for name, df in tables.items():
        logger.info(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return tables


# ── Step 2: Merge Tables ─────────────────────────────────────

def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all tables into a single denormalized DataFrame."""
    logger.info("Merging tables...")

    df = tables["order_items"].copy()

    # Merge orders
    df = df.merge(tables["orders"], on="order_id", how="left")

    # Merge products
    df = df.merge(tables["products"], on="product_id", how="left")

    # Merge customers
    df = df.merge(tables["customers"], on="customer_id", how="left")

    # Merge sellers
    df = df.merge(
        tables["sellers"][["seller_id", "seller_city", "seller_state"]],
        on="seller_id", how="left",
    )

    # Merge category translation
    df = df.merge(
        tables["category_translation"],
        on="product_category_name", how="left",
    )

    # Merge reviews (aggregate per order first)
    review_agg = (
        tables["reviews"]
        .groupby("order_id")["review_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "order_review_score", "count": "order_review_count"})
        .reset_index()
    )
    df = df.merge(review_agg, on="order_id", how="left")

    logger.info(f"  Merged shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── Step 3: Clean ────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean the merged DataFrame."""
    logger.info("Cleaning data...")
    initial_rows = len(df)

    # Keep only delivered orders
    df = df[df["order_status"] == "delivered"].copy()
    logger.info(f"  After status filter: {len(df):,} rows (dropped {initial_rows - len(df):,})")

    # Drop rows without price or category
    df = df.dropna(subset=["price", "product_category_name_english"])
    logger.info(f"  After null drop: {len(df):,} rows")

    # Drop extreme price outliers (< 1 or > 99.5 percentile)
    p995 = df["price"].quantile(0.995)
    df = df[(df["price"] >= 1.0) & (df["price"] <= p995)]
    logger.info(f"  After outlier removal (price <= {p995:.0f}): {len(df):,} rows")

    return df


# ── Step 4: Feature Engineering ──────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 13 features for ML training."""
    logger.info("Engineering features...")

    # ── Numeric features (direct) ─────────────
    df["freight_value"] = df["freight_value"].fillna(0)
    df["product_weight_g"] = df["product_weight_g"].fillna(df["product_weight_g"].median())
    df["product_photos_qty"] = df["product_photos_qty"].fillna(1)
    df["product_description_lenght"] = df["product_description_lenght"].fillna(0)

    # ── Derived: product volume ───────────────
    for col in ["product_length_cm", "product_height_cm", "product_width_cm"]:
        df[col] = df[col].fillna(df[col].median())
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    # ── Review features per product ───────────
    product_reviews = (
        df.groupby("product_id")["order_review_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_review_score", "count": "review_count"})
        .reset_index()
    )
    df = df.drop(columns=["avg_review_score", "review_count"], errors="ignore")
    df = df.merge(product_reviews, on="product_id", how="left")
    df["avg_review_score"] = df["avg_review_score"].fillna(3.0)  # neutral default
    df["review_count"] = df["review_count"].fillna(0)

    # ── Category demand (30-day rolling proxy) ─
    # Use purchase month as proxy since exact 30-day rolling requires sorting
    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    category_demand = (
        df.groupby(["product_category_name_english", "purchase_month"])
        .size()
        .rename("category_demand_30d")
        .reset_index()
    )
    df = df.merge(category_demand, on=["product_category_name_english", "purchase_month"], how="left")
    df["category_demand_30d"] = df["category_demand_30d"].fillna(1)

    # ── Same-state flag ───────────────────────
    df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # ── Label encode categoricals ─────────────
    label_encoders = {}

    le_cat = LabelEncoder()
    df["product_category_encoded"] = le_cat.fit_transform(
        df["product_category_name_english"].astype(str)
    )
    label_encoders["product_category"] = le_cat

    le_seller = LabelEncoder()
    df["seller_state_encoded"] = le_seller.fit_transform(df["seller_state"].astype(str))
    label_encoders["seller_state"] = le_seller

    le_customer = LabelEncoder()
    df["customer_state_encoded"] = le_customer.fit_transform(df["customer_state"].astype(str))
    label_encoders["customer_state"] = le_customer

    # Save label encoders
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    logger.info(f"  Saved label encoders to {LABEL_ENCODERS_PATH}")

    return df


# ── Step 5: Select Final Features ────────────────────────────

FEATURE_COLUMNS = [
    "freight_value",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty",
    "product_description_lenght",
    "product_category_encoded",
    "avg_review_score",
    "review_count",
    "category_demand_30d",
    "seller_state_encoded",
    "customer_state_encoded",
    "is_same_state",
]

TARGET_COLUMN = "price"

# Columns to keep for context (not ML features, but useful for agent)
CONTEXT_COLUMNS = [
    "product_id",
    "product_category_name_english",
    "customer_state",
    "seller_state",
    "order_status",
]


def select_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final feature set and save to parquet."""
    logger.info("Selecting final feature set...")

    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN] + CONTEXT_COLUMNS
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    # Drop any remaining NaN in feature columns
    result = result.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    logger.info(f"  Final dataset: {len(result):,} rows × {len(result.columns)} cols")

    result.to_parquet(FEATURES_PARQUET, index=False)
    logger.info(f"  Saved to {FEATURES_PARQUET}")

    return result


# ── Main ─────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """Execute the full ETL pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DATA PIPELINE")
    logger.info("=" * 60)

    tables = load_raw_tables()
    merged = merge_tables(tables)
    cleaned = clean_data(merged)
    featured = engineer_features(cleaned)
    final = select_and_save(featured)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Output: {FEATURES_PARQUET}")
    logger.info(f"  Shape:  {final.shape}")
    logger.info("=" * 60)
    return final


if __name__ == "__main__":
    run_pipeline()
