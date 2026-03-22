"""
Streamlit Chat UI: Premium negotiation interface with real-time analytics.

Usage:
    streamlit run src/app/streamlit_app.py
"""
import streamlit as st
import requests
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sales Negotiator",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .agent-message {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e0e0e0;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .price-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #0a0a0a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .sidebar .stSelectbox label { color: #ccc; }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
    }

    .stChatInput > div { border-radius: 24px !important; }

    h1, h2, h3 { color: #e8e8e8 !important; }
    p, span, label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)


# ── API Configuration ─────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"


def api_call(endpoint: str, method: str = "GET", data: dict = None) -> dict | None:
    """Make an API call with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server. Run: `uvicorn src.api.main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# ── Session State Initialization ──────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "negotiation_active" not in st.session_state:
    st.session_state.negotiation_active = False
if "current_offer" not in st.session_state:
    st.session_state.current_offer = 0.0
if "negotiation_round" not in st.session_state:
    st.session_state.negotiation_round = 0
if "deal_closed" not in st.session_state:
    st.session_state.deal_closed = False
if "intent_history" not in st.session_state:
    st.session_state.intent_history = []
if "offer_history" not in st.session_state:
    st.session_state.offer_history = []


# ── Sidebar: Product Selection & Metrics ─────────────────────
with st.sidebar:
    st.markdown("## 🛍️ Product Setup")

    # Fetch categories
    categories = api_call("/categories")
    if categories:
        category_list = categories.get("categories", [])
        selected_category = st.selectbox(
            "Product Category",
            options=category_list,
            format_func=lambda x: x.replace("_", " ").title(),
            index=0 if category_list else None,
        )
    else:
        selected_category = "computers_accessories"
        st.text_input("Product Category", value=selected_category, disabled=True)

    # Fetch states
    states_data = api_call("/states")
    if states_data:
        state_list = states_data.get("states", [])
    else:
        state_list = ["SP", "RJ", "MG"]

    col1, col2 = st.columns(2)
    with col1:
        customer_state = st.selectbox("Customer State", state_list, index=0)
    with col2:
        seller_state = st.selectbox("Seller State", state_list, index=0)

    st.markdown("---")

    # Start Negotiation Button
    if st.button("🚀 Start Negotiation", use_container_width=True, type="primary"):
        with st.spinner("Setting up negotiation..."):
            result = api_call("/negotiate/start", method="POST", data={
                "product_category": selected_category,
                "customer_state": customer_state,
                "seller_state": seller_state,
            })
            if result:
                st.session_state.session_id = result["session_id"]
                st.session_state.messages = [
                    {"role": "assistant", "content": result["opening_message"]}
                ]
                st.session_state.negotiation_active = True
                st.session_state.current_offer = result["opening_price"]
                st.session_state.negotiation_round = 1
                st.session_state.deal_closed = False
                st.session_state.intent_history = []
                st.session_state.offer_history = [result["opening_price"]]
                st.rerun()

    # New Negotiation Button
    if st.session_state.negotiation_active:
        if st.button("🔄 New Negotiation", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.negotiation_active = False
            st.session_state.current_offer = 0.0
            st.session_state.negotiation_round = 0
            st.session_state.deal_closed = False
            st.session_state.intent_history = []
            st.session_state.offer_history = []
            st.rerun()

    # ── Live Metrics Panel ────────────────────────────────────
    if st.session_state.negotiation_active:
        st.markdown("---")
        st.markdown("## 📊 Live Metrics")

        st.metric("Current Offer", f"${st.session_state.current_offer:.2f}")
        st.metric("Round", st.session_state.negotiation_round)

        if st.session_state.deal_closed:
            st.success("✅ Deal Closed!")

        # Offer progression chart
        if len(st.session_state.offer_history) > 1:
            st.markdown("### Price Progression")
            st.line_chart(st.session_state.offer_history)

        # Intent distribution
        if st.session_state.intent_history:
            st.markdown("### Intent Distribution")
            intent_counts = {}
            for intent in st.session_state.intent_history:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            st.bar_chart(intent_counts)


# ── Main Chat Area ────────────────────────────────────────────

st.markdown("# 🤝 AI Sales Negotiator")
st.markdown("*Powered by XGBoost Dynamic Pricing + LangGraph Agentic AI*")

if not st.session_state.negotiation_active:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #667eea !important;">Welcome to the Negotiation Arena</h2>
        <p style="color: #999 !important; font-size: 18px;">
            Select a product category in the sidebar and click <b>Start Negotiation</b>
            to begin haggling with our AI sales agent.
        </p>
        <p style="color: #666 !important; font-size: 14px;">
            The agent uses ML-predicted pricing boundaries and cannot drop below its floor price.
            Try to get the best deal you can! 💰
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Type your message... (Try: 'That's too expensive!' or 'Dell offers it for less')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_call("/negotiate/message", method="POST", data={
                    "session_id": st.session_state.session_id,
                    "message": user_input,
                })

            if result:
                response = result["response"]
                st.markdown(response)

                # Update session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.current_offer = result["current_offer"]
                st.session_state.negotiation_round = result["negotiation_round"]
                st.session_state.deal_closed = result["deal_closed"]
                st.session_state.intent_history.append(result["intent"])
                st.session_state.offer_history.append(result["current_offer"])

                st.rerun()
