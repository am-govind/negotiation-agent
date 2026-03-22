---
description: Retrain the ML models after data changes or hyperparameter tuning
---

# Retrain ML Models

// turbo-all

## When to Retrain
- New data added to the `dataset/` folder
- Feature engineering changes in `src/data/pipeline.py`
- Hyperparameter adjustments in `src/ml/train_model.py`

## Steps

1. Re-run the data pipeline (if data or features changed):
```bash
source venv/bin/activate
python -m src.data.pipeline
```

2. Train both models:
```bash
python -m src.ml.train_model
```

3. Verify model performance in the output logs:
   - Price Regressor: R² should be > 0.65, MAE < $50
   - Conversion Classifier: AUC-ROC should be > 0.80

4. Test the price calculator:
```bash
python -c "
from src.ml.price_calculator import get_calculator
calc = get_calculator()
result = calc.get_optimal_price('computers_accessories')
print(f'Target: \${result[\"target_price\"]:.2f}')
print(f'Floor:  \${result[\"floor_price\"]:.2f}')
print(f'Optimal: \${result[\"optimal_price\"]:.2f}')
print(f'Conv Prob: {result[\"optimal_conversion_prob\"]:.2%}')
"
```

## Model Artifacts
After training, these files are updated:
- `models/price_regressor.joblib`
- `models/conversion_classifier.joblib`
- `models/label_encoders.joblib`
- `models/feature_columns.joblib`
