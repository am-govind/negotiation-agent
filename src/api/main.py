"""
FastAPI Backend: Production API serving ML predictions, negotiation agent,
and admin analytics.

API Routes (all under /api/ prefix):
  POST /api/predict-price      — Get ML pricing for a product
  POST /api/negotiate/start    — Start a new negotiation session
  POST /api/negotiate/message  — Send a message in an active session
  GET  /api/categories         — List available product categories (with display names)
  GET  /api/states             — List available states
  GET  /api/health             — Health check

Admin Routes:
  GET  /api/admin/sessions     — List all sessions with metrics
  GET  /api/admin/analytics    — Aggregate stats (win rate, margin, revenue)
  GET  /api/admin/config       — Get current pricing config
  PUT  /api/admin/config       — Update pricing thresholds

Frontend:
  GET  /                       — Customer chat UI
  GET  /admin.html             — Admin dashboard
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    API_HOST, API_PORT, FRONTEND_DIR,
    FLOOR_PRICE_DISCOUNT, DEFAULT_OPENING_MARKUP,
    MAX_NEGOTIATION_ROUNDS, MIN_PROFIT_MARGIN_PCT,
)
from src.ml.price_calculator import get_calculator
from src.agent.state import create_initial_state, NegotiationState
from src.agent.graph import run_negotiation_turn
from src.api.negotiation_logger import get_negotiation_logger
from src.api.auth import (
    LoginRequest, LoginResponse,
    authenticate, require_admin,
)
from src.utils.gemini import extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Session Storage ───────────────────────────────────────────
sessions: dict[str, NegotiationState] = {}

# ── Mutable Runtime Config ────────────────────────────────────
runtime_config = {
    "floor_price_discount": FLOOR_PRICE_DISCOUNT,
    "opening_markup": DEFAULT_OPENING_MARKUP,
    "max_rounds": MAX_NEGOTIATION_ROUNDS,
    "min_profit_margin_pct": MIN_PROFIT_MARGIN_PCT,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models eagerly on startup."""
    logger.info("Loading ML models on startup...")
    try:
        calc = get_calculator()
        logger.info("ML models loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-load models: {e}. They will load on first request.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Dynamic Pricing & Negotiation Agent API",
    description="ML-powered dynamic pricing with agentic negotiation",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────

class PricePredictionRequest(BaseModel):
    product_category: str = Field(..., description="Product category (English name)")
    customer_state: str = Field(default="SP", description="Customer's state code")
    seller_state: str = Field(default="SP", description="Seller's state code")
    freight_value: float | None = None
    product_weight_g: float | None = None
    product_volume_cm3: float | None = None
    product_photos_qty: int = 2
    product_description_length: int = 200
    avg_review_score: float | None = None
    review_count: int | None = None
    category_demand_30d: int | None = None


class PricePredictionResponse(BaseModel):
    target_price: float
    floor_price: float
    optimal_price: float
    optimal_conversion_prob: float
    optimal_expected_value: float
    price_simulations: list[dict]
    product_category: str


class StartNegotiationRequest(BaseModel):
    product_category: str
    customer_state: str = "SP"
    seller_state: str = "SP"


class StartNegotiationResponse(BaseModel):
    session_id: str
    opening_message: str
    opening_price: float
    product_category: str
    image_url: str


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    response: str
    current_offer: float
    negotiation_round: int
    deal_closed: bool
    deal_abandoned: bool
    intent: str
    value_adds_offered: list[str]


class ConfigUpdateRequest(BaseModel):
    floor_price_discount: float | None = None
    opening_markup: float | None = None
    max_rounds: int | None = None
    min_profit_margin_pct: float | None = None


# ── Core API Endpoints ────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    from src.utils.llm import get_provider_info
    return {"status": "healthy", "version": "2.1.0", "llm": get_provider_info()}


@app.get("/api/categories")
async def list_categories():
    """List all product categories with display names and avg prices."""
    try:
        calc = get_calculator()
        categories = calc.get_category_display_info()
        return {"categories": categories, "count": len(categories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/states")
async def list_states():
    """List all available customer/seller states."""
    try:
        calc = get_calculator()
        states = calc.get_available_states()
        # Map state codes to display names
        state_names = {
            "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
            "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
            "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
            "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
            "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí",
            "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
            "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
            "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
        }
        state_list = [
            {"code": s, "name": state_names.get(s, s)}
            for s in states
        ]
        return {"states": state_list, "count": len(state_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-price", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """Get ML pricing predictions for a product."""
    try:
        calc = get_calculator()
        result = calc.get_optimal_price(
            product_category=request.product_category,
            customer_state=request.customer_state,
            seller_state=request.seller_state,
            freight_value=request.freight_value,
            product_weight_g=request.product_weight_g,
            product_volume_cm3=request.product_volume_cm3,
            product_photos_qty=request.product_photos_qty,
            product_description_length=request.product_description_length,
            avg_review_score=request.avg_review_score,
            review_count=request.review_count,
            category_demand_30d=request.category_demand_30d,
        )
        return PricePredictionResponse(**result)
    except Exception as e:
        logger.error(f"Price prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/negotiate/start", response_model=StartNegotiationResponse)
async def start_negotiation(request: StartNegotiationRequest):
    """Start a new negotiation session."""
    try:
        calc = get_calculator()
        pricing = calc.get_optimal_price(
            product_category=request.product_category,
            customer_state=request.customer_state,
            seller_state=request.seller_state,
        )

        session_id = str(uuid.uuid4())
        state = create_initial_state(
            target_price=pricing["target_price"],
            floor_price=pricing["floor_price"],
            optimal_price=pricing["optimal_price"],
            product_category=request.product_category,
            customer_state=request.customer_state,
            seller_state=request.seller_state,
            price_simulations=pricing["price_simulations"],
            conversion_probability=pricing["optimal_conversion_prob"],
        )

        state, opening_response = run_negotiation_turn(
            state,
            f"Hi, I'm interested in buying a {request.product_category.replace('_', ' ')}.",
        )

        sessions[session_id] = state

        from src.utils.images import get_category_image

        return StartNegotiationResponse(
            session_id=session_id,
            opening_message=opening_response,
            opening_price=pricing["optimal_price"] * DEFAULT_OPENING_MARKUP,
            product_category=request.product_category,
            image_url=get_category_image(request.product_category),
        )

    except Exception as e:
        logger.error(f"Failed to start negotiation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/negotiate/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message in an active negotiation session."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = sessions[request.session_id]

        if state.get("deal_closed") or state.get("deal_abandoned"):
            # Log completed negotiation
            _log_session(request.session_id, state)
            return MessageResponse(
                session_id=request.session_id,
                response="This negotiation has already concluded. Start a new session to negotiate again.",
                current_offer=state["current_offer"],
                negotiation_round=state["negotiation_round"],
                deal_closed=state.get("deal_closed", False),
                deal_abandoned=state.get("deal_abandoned", False),
                intent=state.get("intent", ""),
                value_adds_offered=state.get("value_adds_offered", []),
            )

        updated_state, response_text = run_negotiation_turn(state, request.message)
        sessions[request.session_id] = updated_state

        # Log if negotiation is now complete
        if updated_state.get("deal_closed") or updated_state.get("deal_abandoned"):
            _log_session(request.session_id, updated_state)

        return MessageResponse(
            session_id=request.session_id,
            response=response_text,
            current_offer=updated_state["current_offer"],
            negotiation_round=updated_state["negotiation_round"],
            deal_closed=updated_state.get("deal_closed", False),
            deal_abandoned=updated_state.get("deal_abandoned", False),
            intent=updated_state.get("intent", ""),
            value_adds_offered=updated_state.get("value_adds_offered", []),
        )

    except Exception as e:
        logger.error(f"Negotiation turn failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/negotiate/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current state of a negotiation session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]
    return {
        "session_id": session_id,
        "product_category": state["product_category"],
        "current_offer": state["current_offer"],
        "target_price": state["target_price"],
        "negotiation_round": state["negotiation_round"],
        "deal_closed": state.get("deal_closed", False),
        "deal_abandoned": state.get("deal_abandoned", False),
        "intent": state.get("intent", ""),
        "value_adds_offered": state.get("value_adds_offered", []),
        "margin_retained": _calc_margin(state),
    }


# ── Auth Endpoints ────────────────────────────────────────────

@app.post("/api/admin/login", response_model=LoginResponse)
async def admin_login(request: LoginRequest):
    """Authenticate admin and return a JWT token."""
    token = authenticate(request.username, request.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return LoginResponse(token=token, username=request.username, expires_in=86400)


# ── Admin Endpoints (protected) ───────────────────────────────

@app.get("/api/admin/sessions")
async def admin_list_sessions(admin=Depends(require_admin)):
    """List all active sessions with metrics."""
    active = []
    for sid, state in sessions.items():
        active.append({
            "session_id": sid,
            "product_category": state["product_category"],
            "current_offer": state["current_offer"],
            "target_price": state["target_price"],
            "floor_price": state["floor_price"],
            "negotiation_round": state["negotiation_round"],
            "deal_closed": state.get("deal_closed", False),
            "deal_abandoned": state.get("deal_abandoned", False),
            "margin_retained": _calc_margin(state),
        })
    return {"active_sessions": active, "count": len(active)}


@app.get("/api/admin/sessions/{session_id}")
async def admin_get_session(session_id: str, admin=Depends(require_admin)):
    """Get the full message history of a specific session."""
    # First check completed logs
    nl = get_negotiation_logger()
    log_file = nl._log_dir / f"{session_id}.json"
    if log_file.exists():
        import json
        with open(log_file, "r") as f:
            return json.load(f)
            
    # Then check active in-memory sessions
    if session_id in sessions:
        from src.utils.gemini import extract_text as _et
        msgs = []
        for m in sessions[session_id].get("messages", []):
            role = m.type if hasattr(m, "type") else m.get("role", "unknown")
            if role == "system": continue # Skip internal prompts
            content = _et(m.content) if hasattr(m, "content") else str(m)
            msgs.append({"role": role, "content": content})
        return {"session_id": session_id, "messages": msgs, "status": "active"}
        
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/admin/analytics")
async def admin_analytics(admin=Depends(require_admin)):
    """Aggregate analytics from all logged sessions."""
    try:
        nl = get_negotiation_logger()
        return nl.get_analytics()
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/config")
async def admin_get_config(admin=Depends(require_admin)):
    """Get current runtime pricing configuration."""
    return runtime_config.copy()


@app.put("/api/admin/config")
async def admin_update_config(request: ConfigUpdateRequest, admin=Depends(require_admin)):
    """Update runtime pricing configuration."""
    if request.floor_price_discount is not None:
        runtime_config["floor_price_discount"] = request.floor_price_discount
    if request.opening_markup is not None:
        runtime_config["opening_markup"] = request.opening_markup
    if request.max_rounds is not None:
        runtime_config["max_rounds"] = request.max_rounds
    if request.min_profit_margin_pct is not None:
        runtime_config["min_profit_margin_pct"] = request.min_profit_margin_pct
    logger.info(f"Config updated: {runtime_config}")
    return runtime_config


# ── Helpers ───────────────────────────────────────────────────

def _calc_margin(state: NegotiationState) -> float:
    """Calculate percentage of margin retained."""
    target = state["target_price"]
    floor = state["floor_price"]
    current = state["current_offer"]
    if target == floor:
        return 100.0
    return round(max(0, min(100, ((current - floor) / (target - floor)) * 100)), 1)


def _log_session(session_id: str, state: NegotiationState):
    """Log a negotiation session to disk."""
    try:
        from src.utils.gemini import extract_text as _et
        messages = []
        for msg in state.get("messages", []):
            role = "assistant"
            if hasattr(msg, "type"):
                role = msg.type
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
            content = _et(msg.content) if hasattr(msg, "content") else str(msg)
            messages.append({"role": role, "content": content})

        outcome = "closed" if state.get("deal_closed") else "abandoned" if state.get("deal_abandoned") else "active"
        nl = get_negotiation_logger()
        nl.log_session(session_id, state, messages, outcome)
    except Exception as e:
        logger.warning(f"Could not log session {session_id}: {e}")


# ── Serve Frontend Static Files ──────────────────────────────
# Mount AFTER all API routes so /api/* takes priority
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"Serving frontend from {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend not found at {FRONTEND_DIR}. Run: cd frontend && npm run build")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
