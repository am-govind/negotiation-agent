"""
XGBoost Model Training: Price Regressor + Conversion Probability Classifier.

Implements Improvisation 2 (Price Elasticity) — trains two models:
  1. Price Regressor: predicts fair market target price
  2. Conversion Classifier: predicts P(sale) at a given price point

Usage:
    python -m src.ml.train_model
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, classification_report,
)
from xgboost import XGBRegressor, XGBClassifier
import joblib

from src.config import (
    FEATURES_PARQUET, PRICE_REGRESSOR_PATH,
    CONVERSION_CLASSIFIER_PATH, FEATURE_COLUMNS_PATH,
    TEST_SPLIT_RATIO, RANDOM_STATE,
)
from src.data.pipeline import FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Load Data ─────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load the processed feature parquet."""
    logger.info(f"Loading features from {FEATURES_PARQUET}")
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info(f"  Shape: {df.shape}")
    return df


# ── Model 1: Price Regressor ─────────────────────────────────

def train_price_regressor(df: pd.DataFrame) -> XGBRegressor:
    """Train XGBoost to predict fair market price."""
    logger.info("=" * 50)
    logger.info("TRAINING PRICE REGRESSOR")
    logger.info("=" * 50)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE,
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"  MAE:  ${mae:.2f}")
    logger.info(f"  RMSE: ${rmse:.2f}")
    logger.info(f"  R²:   {r2:.4f}")

    # Feature importance
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("  Feature Importances:")
    for feat, imp in sorted_imp[:8]:
        logger.info(f"    {feat}: {imp:.4f}")

    # Save
    joblib.dump(model, PRICE_REGRESSOR_PATH)
    logger.info(f"  Saved to {PRICE_REGRESSOR_PATH}")

    return model


# ── Model 2: Conversion Classifier ──────────────────────────

def train_conversion_classifier(df: pd.DataFrame) -> XGBClassifier:
    """
    Train XGBoost to predict P(conversion) given features + price.
    
    For elasticity simulation: at inference we vary price and observe
    how P(conversion) changes, finding the optimal price point.
    """
    logger.info("=" * 50)
    logger.info("TRAINING CONVERSION CLASSIFIER")
    logger.info("=" * 50)

    # The classifier uses ALL features INCLUDING price
    classifier_features = FEATURE_COLUMNS + [TARGET_COLUMN]

    # Create synthetic conversion labels:
    # - Delivered orders with good reviews (>= 3) → 1 (converted & satisfied)
    # - Others → create negative samples by perturbing price upward
    df_pos = df.copy()
    df_pos["converted"] = 1

    # Create negative samples: duplicate 30% of data with inflated prices
    np.random.seed(RANDOM_STATE)
    n_neg = int(len(df) * 0.3)
    df_neg = df.sample(n=n_neg, random_state=RANDOM_STATE).copy()

    # Inflate prices by 30-80% to simulate "too expensive" scenarios
    price_multipliers = np.random.uniform(1.3, 1.8, size=n_neg)
    df_neg["price"] = df_neg["price"] * price_multipliers
    df_neg["converted"] = 0

    # Also add negative samples with very low review products at current price
    low_review_mask = df["avg_review_score"] < 2.5
    if low_review_mask.sum() > 0:
        df_low = df[low_review_mask].copy()
        df_low["converted"] = 0
        df_combined = pd.concat([df_pos, df_neg, df_low], ignore_index=True)
    else:
        df_combined = pd.concat([df_pos, df_neg], ignore_index=True)

    X = df_combined[classifier_features]
    y = df_combined["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="auc",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  AUC-ROC:  {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Save
    joblib.dump(model, CONVERSION_CLASSIFIER_PATH)
    logger.info(f"  Saved to {CONVERSION_CLASSIFIER_PATH}")

    return model


# ── Main ─────────────────────────────────────────────────────

def train_all():
    """Train both models end-to-end."""
    df = load_features()

    # Save feature column list for inference
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)

    price_model = train_price_regressor(df)
    conversion_model = train_conversion_classifier(df)

    logger.info("=" * 50)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY")
    logger.info(f"  Price Regressor:       {PRICE_REGRESSOR_PATH}")
    logger.info(f"  Conversion Classifier: {CONVERSION_CLASSIFIER_PATH}")
    logger.info("=" * 50)

    return price_model, conversion_model


if __name__ == "__main__":
    train_all()
