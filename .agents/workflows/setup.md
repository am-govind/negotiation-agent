---
description: Set up the development environment and install all dependencies
---

# Setup Development Environment

// turbo-all

## Steps

1. Create and activate the Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the environment template and add your API key:
```bash
cp .env.example .env
```

4. Edit `.env` and set your Gemini API key:
```
GOOGLE_API_KEY=your-actual-gemini-api-key
```

5. Run the data pipeline to process the Olist dataset:
```bash
python -m src.data.pipeline
```

6. Train the ML models:
```bash
python -m src.ml.train_model
```

## Verification

After setup, you should have:
- `data/processed/features.parquet` — 108K+ rows of engineered features
- `models/price_regressor.joblib` — XGBoost price model (R² ≈ 0.73)
- `models/conversion_classifier.joblib` — XGBoost conversion model (AUC ≈ 0.86)
- `models/label_encoders.joblib` — Fitted label encoders
- `models/feature_columns.joblib` — Feature column list
