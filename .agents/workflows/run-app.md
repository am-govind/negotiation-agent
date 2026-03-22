---
description: Run the negotiation chat application (API + Streamlit frontend)
---

# Run the Application

## Prerequisites
- Environment set up (run `/setup` workflow first)
- `GOOGLE_API_KEY` set in `.env` or environment

## Steps

1. Start the FastAPI backend server (Terminal 1):
```bash
source venv/bin/activate
uvicorn src.api.main:app --port 8000 --reload
```

2. Launch the Streamlit chat UI (Terminal 2):
```bash
source venv/bin/activate
streamlit run src/app/streamlit_app.py
```

3. Open the Streamlit URL shown in Terminal 2 (usually http://localhost:8501)

4. In the sidebar:
   - Select a product category
   - Choose customer/seller states
   - Click **🚀 Start Negotiation**

5. Start haggling with the AI agent!

## Tips
- Try saying: "That's too expensive!" to trigger price negotiation
- Mention "Dell" or "Amazon" to test competitor differentiation
- Ask "What about shipping?" to test the shipping skill
- Say "I'll take it" to close the deal
