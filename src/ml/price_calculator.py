"""
Price Calculator: Inference utilities for the trained ML models.

Provides:
  - get_target_price(features) → float
  - get_floor_price(target) → float
  - get_optimal_price(features) → dict  (elasticity simulation)
  - get_pricing_context(product_category, customer_state, ...) → dict
"""
import logging
import numpy as np
import pandas as pd
import joblib

from src.config import (
    PRICE_REGRESSOR_PATH, CONVERSION_CLASSIFIER_PATH,
    LABEL_ENCODERS_PATH, FEATURE_COLUMNS_PATH,
    FLOOR_PRICE_DISCOUNT, ELASTICITY_PRICE_POINTS,
    FEATURES_PARQUET,
)

logger = logging.getLogger(__name__)


class PriceCalculator:
    """Production-grade price calculator backed by trained XGBoost models."""

    def __init__(self, eager: bool = True):
        self._price_model = None
        self._conversion_model = None
        self._label_encoders = None
        self._feature_columns = None
        self._category_stats = None
        if eager:
            self._ensure_loaded()

    def _ensure_loaded(self):
        """Lazy-load models and encoders on first use."""
        if self._price_model is None:
            logger.info("Loading ML models...")
            self._price_model = joblib.load(PRICE_REGRESSOR_PATH)
            self._conversion_model = joblib.load(CONVERSION_CLASSIFIER_PATH)
            self._label_encoders = joblib.load(LABEL_ENCODERS_PATH)
            self._feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            self._load_category_stats()

    def _load_category_stats(self):
        """Pre-compute category-level statistics for defaults."""
        try:
            df = pd.read_parquet(FEATURES_PARQUET)
            self._category_stats = (
                df.groupby("product_category_name_english")
                .agg({
                    "price": ["mean", "median", "min", "max"],
                    "freight_value": "mean",
                    "product_weight_g": "median",
                    "product_volume_cm3": "median",
                    "avg_review_score": "mean",
                    "review_count": "median",
                    "category_demand_30d": "median",
                })
            )
            # Flatten multi-level columns
            self._category_stats.columns = [
                f"{c[0]}_{c[1]}" for c in self._category_stats.columns
            ]
            logger.info(f"  Loaded stats for {len(self._category_stats)} categories")
        except Exception as e:
            logger.warning(f"Could not load category stats: {e}")
            self._category_stats = None

    def get_available_categories(self) -> list[str]:
        """Return list of product categories the model knows about."""
        self._ensure_loaded()
        return list(self._label_encoders["product_category"].classes_)

    def get_available_states(self) -> list[str]:
        """Return list of states the model knows about."""
        self._ensure_loaded()
        return list(self._label_encoders["customer_state"].classes_)

    def get_category_display_info(self) -> list[dict]:
        """Return human-readable category info with display names and avg prices."""
        self._ensure_loaded()
        categories = list(self._label_encoders["product_category"].classes_)
        result = []
        for cat in categories:
            display_name = cat.replace("_", " ").title()
            avg_price = 0.0
            if self._category_stats is not None and cat in self._category_stats.index:
                avg_price = round(float(self._category_stats.loc[cat].get("price_mean", 0)), 2)
            result.append({
                "id": cat,
                "name": display_name,
                "avg_price": avg_price,
            })
        return result

    @staticmethod
    def get_profit_metrics(target_price: float, floor_price: float, final_price: float) -> dict:
        """Calculate profit metrics for a completed/in-progress negotiation."""
        margin_range = target_price - floor_price
        margin_retained_pct = (
            round(((final_price - floor_price) / margin_range) * 100, 1)
            if margin_range > 0 else 100.0
        )
        discount_pct = round(((target_price - final_price) / target_price) * 100, 1) if target_price > 0 else 0.0
        estimated_profit = round(final_price - floor_price, 2)
        return {
            "margin_retained_pct": max(0, min(100, margin_retained_pct)),
            "discount_given_pct": max(0, discount_pct),
            "estimated_profit": max(0, estimated_profit),
        }

    def build_features(
        self,
        product_category: str,
        customer_state: str = "SP",
        seller_state: str = "SP",
        freight_value: float | None = None,
        product_weight_g: float | None = None,
        product_volume_cm3: float | None = None,
        product_photos_qty: int = 2,
        product_description_length: int = 200,
        avg_review_score: float | None = None,
        review_count: int | None = None,
        category_demand_30d: int | None = None,
    ) -> pd.DataFrame:
        """Build a feature DataFrame from user-provided inputs, filling defaults from category stats."""
        self._ensure_loaded()

        # Get category defaults
        defaults = {}
        if self._category_stats is not None and product_category in self._category_stats.index:
            cat_row = self._category_stats.loc[product_category]
            defaults = {
                "freight_value": cat_row.get("freight_value_mean", 20.0),
                "product_weight_g": cat_row.get("product_weight_g_median", 1000.0),
                "product_volume_cm3": cat_row.get("product_volume_cm3_median", 5000.0),
                "avg_review_score": cat_row.get("avg_review_score_mean", 4.0),
                "review_count": cat_row.get("review_count_median", 5),
                "category_demand_30d": cat_row.get("category_demand_30d_median", 50),
            }

        # Encode categoricals
        cat_encoded = self._safe_encode("product_category", product_category)
        seller_encoded = self._safe_encode("seller_state", seller_state)
        customer_encoded = self._safe_encode("customer_state", customer_state)

        features = {
            "freight_value": freight_value or defaults.get("freight_value", 20.0),
            "product_weight_g": product_weight_g or defaults.get("product_weight_g", 1000.0),
            "product_volume_cm3": product_volume_cm3 or defaults.get("product_volume_cm3", 5000.0),
            "product_photos_qty": product_photos_qty,
            "product_description_lenght": product_description_length,
            "product_category_encoded": cat_encoded,
            "avg_review_score": avg_review_score or defaults.get("avg_review_score", 4.0),
            "review_count": review_count if review_count is not None else defaults.get("review_count", 5),
            "category_demand_30d": category_demand_30d if category_demand_30d is not None else defaults.get("category_demand_30d", 50),
            "seller_state_encoded": seller_encoded,
            "customer_state_encoded": customer_encoded,
            "is_same_state": int(customer_state == seller_state),
        }

        return pd.DataFrame([features])[self._feature_columns]

    def _safe_encode(self, encoder_name: str, value: str) -> int:
        """Encode a categorical value, falling back to 0 for unknown values."""
        le = self._label_encoders[encoder_name]
        if value in le.classes_:
            return int(le.transform([value])[0])
        logger.warning(f"Unknown {encoder_name} value: {value}, using default 0")
        return 0

    def get_target_price(self, features: pd.DataFrame) -> float:
        """Predict the fair market price for given features."""
        self._ensure_loaded()
        prediction = self._price_model.predict(features)[0]
        return round(max(float(prediction), 1.0), 2)

    def get_floor_price(self, target_price: float) -> float:
        """Calculate the absolute minimum acceptable price."""
        return round(target_price * FLOOR_PRICE_DISCOUNT, 2)

    def get_conversion_probability(self, features: pd.DataFrame, price: float) -> float:
        """Predict P(conversion) at a specific price point."""
        self._ensure_loaded()
        # Classifier uses features + price
        features_with_price = features.copy()
        features_with_price["price"] = price
        classifier_cols = self._feature_columns + ["price"]
        proba = self._conversion_model.predict_proba(features_with_price[classifier_cols])[0][1]
        return round(float(proba), 4)

    def get_optimal_price(
        self,
        product_category: str,
        customer_state: str = "SP",
        seller_state: str = "SP",
        **kwargs,
    ) -> dict:
        """
        Run price elasticity simulation (Improvisation 2).
        
        Tests N price points around the predicted target and returns
        the one maximizing Expected Value = Price × P(conversion).
        """
        self._ensure_loaded()

        features = self.build_features(
            product_category=product_category,
            customer_state=customer_state,
            seller_state=seller_state,
            **kwargs,
        )

        target_price = self.get_target_price(features)
        floor_price = self.get_floor_price(target_price)

        # Simulate price points: from floor to 20% above target
        price_points = np.linspace(
            floor_price,
            target_price * 1.20,
            ELASTICITY_PRICE_POINTS,
        )

        results = []
        for p in price_points:
            conv_prob = self.get_conversion_probability(features, p)
            expected_value = p * conv_prob
            results.append({
                "price": round(float(p), 2),
                "conversion_probability": conv_prob,
                "expected_value": round(float(expected_value), 2),
            })

        # Find optimal (max expected value)
        best = max(results, key=lambda r: r["expected_value"])

        return {
            "target_price": target_price,
            "floor_price": floor_price,
            "optimal_price": best["price"],
            "optimal_conversion_prob": best["conversion_probability"],
            "optimal_expected_value": best["expected_value"],
            "price_simulations": results,
            "product_category": product_category,
            "customer_state": customer_state,
        }


# ── Module-level singleton ────────────────────────────────────
_calculator: PriceCalculator | None = None


def get_calculator() -> PriceCalculator:
    """Get or create the global PriceCalculator singleton."""
    global _calculator
    if _calculator is None:
        _calculator = PriceCalculator()
    return _calculator
