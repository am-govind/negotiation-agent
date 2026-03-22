---
description: Run the multi-agent arena to evaluate the negotiation agent against buyer personas
---

# Run Arena Evaluation

## Prerequisites
- Environment set up (run `/setup` workflow first)
- `GOOGLE_API_KEY` set in `.env` or environment
- ML models trained

## Steps

1. Run the arena with all 3 buyer personas (10 runs each):
```bash
source venv/bin/activate
python -m src.evaluation.arena --runs 10 --personas aggressive value urgent
```

2. For a quick test run (3 runs, single persona):
```bash
python -m src.evaluation.arena --runs 3 --personas aggressive
```

3. For a specific product category:
```bash
python -m src.evaluation.arena --runs 10 --categories computers_accessories health_beauty
```

4. View the dashboard:
```bash
streamlit run src/app/dashboard.py
```

## Metrics Tracked
- **Win Rate**: % of negotiations resulting in a deal
- **Margin Retained**: How close the final price is to target vs floor
- **Avg Rounds**: Number of back-and-forth messages to close
- **Avg Discount**: % discount given from opening price

## Output
Results are saved to `outputs/arena/arena_results_YYYYMMDD_HHMMSS.csv`
