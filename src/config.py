"""
Project-wide configuration and path constants.
"""
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so GOOGLE_API_KEY is available
load_dotenv()

# ── Project Root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
ARENA_DIR = OUTPUTS_DIR / "arena"
NEGOTIATION_LOG_DIR = OUTPUTS_DIR / "negotiations"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

# ── Raw CSV File Paths ────────────────────────────────────────
ORDER_ITEMS_CSV = RAW_DATA_DIR / "olist_order_items_dataset.csv"
ORDERS_CSV = RAW_DATA_DIR / "olist_orders_dataset.csv"
PRODUCTS_CSV = RAW_DATA_DIR / "olist_products_dataset.csv"
CUSTOMERS_CSV = RAW_DATA_DIR / "olist_customers_dataset.csv"
REVIEWS_CSV = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
SELLERS_CSV = RAW_DATA_DIR / "olist_sellers_dataset.csv"
GEOLOCATION_CSV = RAW_DATA_DIR / "olist_geolocation_dataset.csv"
PAYMENTS_CSV = RAW_DATA_DIR / "olist_order_payments_dataset.csv"
CATEGORY_TRANSLATION_CSV = RAW_DATA_DIR / "product_category_name_translation.csv"

# ── Processed File Paths ──────────────────────────────────────
FEATURES_PARQUET = PROCESSED_DATA_DIR / "features.parquet"

# ── Model Artifact Paths ──────────────────────────────────────
PRICE_REGRESSOR_PATH = MODELS_DIR / "price_regressor.joblib"
CONVERSION_CLASSIFIER_PATH = MODELS_DIR / "conversion_classifier.joblib"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"

# ── ML Config ─────────────────────────────────────────────────
FLOOR_PRICE_DISCOUNT = 0.85          # Floor = target × 0.85
ELASTICITY_PRICE_POINTS = 5          # Number of price simulations
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# ── Negotiation Config ────────────────────────────────────────
MAX_NEGOTIATION_ROUNDS = 10
DEFAULT_OPENING_MARKUP = 1.10        # Start 10% above target
MIN_PROFIT_MARGIN_PCT = 15           # Agent's minimum profit margin target

# ── API Config ────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Admin Auth Config ─────────────────────────────────────────
import os
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# ── Ensure directories exist ─────────────────────────────────
for d in [PROCESSED_DATA_DIR, MODELS_DIR, EDA_DIR, ARENA_DIR, NEGOTIATION_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
