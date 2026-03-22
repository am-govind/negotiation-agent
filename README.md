# 🤝 AI Sales Negotiation Agent

An industry-grade AI system combining **XGBoost dynamic pricing** with an **LLM-powered negotiation agent**. Features deterministic skill execution, intent routing, profit maximization, and a premium dark-themed frontend with admin analytics.

---

## ✨ Key Features

- **ML-Driven Pricing** — XGBoost models predict optimal prices and conversion probability from 100K+ real e-commerce orders
- **Agentic Negotiation** — LangGraph state machine with 5-node pipeline: Router → Skill Selector → Agentic Core → Tool Executor → Generator
- **Deterministic Skills** — Python skill registry for pricing, shipping, competitor analysis, and deal closing (not hallucinated)
- **Profit Maximization** — Agent prioritizes selling at or above optimal price, with configurable margins
- **Intent Routing** — 8 customer intents classified in real-time for tailored negotiation strategies
- **Admin Dashboard & Logs** — JWT-protected analytics: win rate, revenue, margin retention, and full chat session history
- **Rich Chat UI** — Real-time Unsplash product images, Markdown rendering, and dynamic pricing updates
- **Multi-Agent Arena** — Automated buyer personas (Aggressive, Value Seeker, Urgent) for agent evaluation
- **Dual LLM Support** — HuggingFace (primary) + Gemini 2.5 Flash (fallback/alternative), switchable via `.env`

---

## 🏗️ Architecture

```
Customer Message
  → Intent Router (lightweight LLM, JSON classification)
  → Skill Selector (deterministic Python: pricing, shipping, competitor data)
  → Agentic Core (LLM reasoning with skill results + tools)
  → Tool Executor (validates offers against floor price)
  → Response Generator (conversational reply)
  → Customer
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Engine | XGBoost, Pandas, scikit-learn |
| Agent Framework | LangGraph, LangChain |
| LLM (Primary) | Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API |
| LLM (Backup) | meta-llama/Llama-3.1-70B-Instruct |
| LLM (Fallback) | Gemini 2.5 Flash (Google) |
| API | FastAPI + JWT Authentication |
| Frontend | Vite + Vanilla JS (dark glassmorphism theme) |
| Dataset | Olist Brazilian E-Commerce (100K+ orders) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend build)
- A [HuggingFace API token](https://huggingface.co/settings/tokens) (free) **or** a Google Gemini API key

### 1. Clone & Create Virtual Environment

```bash
git clone <repo-url>
cd dynamic-prcining-n-negotiation-agent

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# HuggingFace (Primary — recommended)
HF_API_TOKEN=hf_your_token_here
HF_MODEL=Qwen/Qwen2.5-72B-Instruct
HF_BACKUP_MODEL=meta-llama/Llama-3.1-70B-Instruct

# Google Gemini (Fallback — used if HF_API_TOKEN is not set)
GOOGLE_API_KEY=your_gemini_key_here

# Unsplash API (Optional — for product images in chat)
UNSPLASH_ACCESS_KEY=your_unsplash_access_key

# Admin Dashboard Login
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
JWT_SECRET=change-this-to-a-random-string
```

> **Provider priority:** If `HF_API_TOKEN` is set → HuggingFace. If HF fails or is not set → auto-fallback to Gemini.

### 4. Run Data Pipeline

```bash
python -m src.data.pipeline
```

This processes the raw Olist CSVs into feature-engineered parquet files.

### 5. Train ML Models

```bash
python -m src.ml.train_model
```

Trains the XGBoost price regressor and conversion classifier. Models are saved to `models/`.

### 6. Build Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

Produces optimized static files in `frontend/dist/`.

### 7. Start the Server

```bash
uvicorn src.api.main:app --port 8000 --reload
```

### 8. Open in Browser

| Page | URL |
|------|-----|
| 🛍️ **Customer Chat** | http://localhost:8000 |
| 🔐 **Admin Login** | http://localhost:8000/login.html |
| 📊 **Admin Dashboard** | http://localhost:8000/admin.html (requires login) |

### 9. Run Arena Evaluation (Optional)

```bash
python -m src.evaluation.arena --runs 10 --personas aggressive value urgent
```

---

## 🔧 Configuration

All configuration is via environment variables (`.env`) and `src/config.py`.

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_API_TOKEN` | — | HuggingFace API token (enables HF models) |
| `HF_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Primary HuggingFace model |
| `HF_BACKUP_MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | Backup model if primary fails |
| `GOOGLE_API_KEY` | — | Gemini API key (fallback) |
| `UNSPLASH_ACCESS_KEY` | — | Unsplash API key for dynamic product images |

### Negotiation Tuning

| Variable (in `config.py`) | Default | Description |
|---------------------------|---------|-------------|
| `FLOOR_PRICE_DISCOUNT` | `0.85` | Floor price = target × 0.85 |
| `DEFAULT_OPENING_MARKUP` | `1.10` | Opening offer = target × 1.10 |
| `MAX_NEGOTIATION_ROUNDS` | `10` | Max rounds before auto-close |
| `MIN_PROFIT_MARGIN_PCT` | `15` | Minimum profit margin target (%) |

### Admin Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_USERNAME` | `admin` | Dashboard login username |
| `ADMIN_PASSWORD` | `admin123` | Dashboard login password |
| `JWT_SECRET` | — | Secret key for JWT tokens |

---

## 📁 Project Structure

```
├── dataset/                     # Raw Olist CSVs (9 files)
├── data/processed/              # Feature-engineered parquet
├── models/                      # Trained XGBoost artifacts (.joblib)
├── outputs/
│   ├── arena/                   # Arena simulation results
│   ├── eda/                     # EDA charts
│   └── negotiations/            # Logged negotiation sessions (JSON)
├── frontend/
│   ├── index.html               # Customer chat page
│   ├── login.html               # Admin login page
│   ├── admin.html               # Admin dashboard page
│   ├── src/
│   │   ├── main.js              # Chat application logic
│   │   ├── admin.js             # Dashboard logic
│   │   ├── login.js             # Login logic
│   │   ├── api.js               # Shared API client (with JWT auth)
│   │   └── styles/              # CSS design system
│   ├── vite.config.js           # Multi-page Vite build config
│   └── dist/                    # Production build output
├── src/
│   ├── config.py                # Central configuration & constants
│   ├── data/pipeline.py         # ETL + feature engineering
│   ├── ml/
│   │   ├── train_model.py       # XGBoost training pipeline
│   │   └── price_calculator.py  # Inference, elasticity, profit metrics
│   ├── api/
│   │   ├── main.py              # FastAPI server (all /api/* routes)
│   │   ├── auth.py              # JWT authentication
│   │   ├── negotiation_logger.py # Session persistence + analytics
│   │   └── tools.py             # Strict tool calling (offer validation)
│   ├── agent/
│   │   ├── state.py             # NegotiationState TypedDict
│   │   ├── router.py            # Intent classification (8 intents)
│   │   ├── graph.py             # LangGraph 5-node pipeline
│   │   ├── prompts.py           # Intent-specific prompt templates
│   │   └── skills/              # Deterministic skill modules
│   │       ├── registry.py      # Skill registry & discovery
│   │       ├── pricing.py       # ML-backed pricing skill
│   │       ├── shipping.py      # Shipping & logistics
│   │       ├── competitor.py    # Competitor analysis
│   │       └── ...
│   ├── utils/
│   │   ├── llm.py               # Centralized LLM factory (HF + Gemini)
│   │   └── gemini.py            # Response parsing (extract_text, extract_json)
│   └── evaluation/
│       ├── arena.py             # Multi-agent simulation harness
│       ├── buyer_personas.py    # 3 buyer personas
│       └── metrics.py           # Evaluation metrics
├── .agents/workflows/           # Runnable workflow definitions
├── .env                         # Environment variables
├── .env.example                 # Template for .env
└── requirements.txt             # Python dependencies
```

---

## 🔌 API Reference

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + active LLM provider info |
| `GET` | `/api/categories` | List product categories with display names & avg prices |
| `GET` | `/api/states` | List Brazilian states with full names |
| `POST` | `/api/predict-price` | Get ML pricing for a product configuration |
| `POST` | `/api/negotiate/start` | Start a new negotiation session |
| `POST` | `/api/negotiate/message` | Send a message in an active session |
| `GET` | `/api/negotiate/{id}/status` | Get session status |

### Admin Endpoints (JWT required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/login` | Authenticate and get JWT token |
| `GET` | `/api/admin/sessions` | List all active sessions with metrics |
| `GET` | `/api/admin/analytics` | Aggregate stats (win rate, margin, revenue) |
| `GET` | `/api/admin/config` | Get runtime pricing configuration |
| `PUT` | `/api/admin/config` | Update pricing thresholds |

---

## 🧪 Running Tests

```bash
# Health check
curl http://localhost:8000/api/health

# Check LLM provider
curl -s http://localhost:8000/api/health | python3 -m json.tool

# Test intent routing
curl -X POST http://localhost:8000/api/negotiate/start \
  -H 'Content-Type: application/json' \
  -d '{"product_category": "computers_accessories", "customer_state": "SP", "seller_state": "SP"}'

# Test admin login
curl -X POST http://localhost:8000/api/admin/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "admin123"}'
```

---

## 📊 ML Models

Two XGBoost models trained on 100K+ Olist e-commerce orders:

1. **Price Regressor** — Predicts optimal price given product features, category, region, and demand
2. **Conversion Classifier** — Predicts purchase probability at a given price point

**Price Elasticity Simulation:** Evaluates 5 price points per product, selecting the one that maximizes `Price × P(conversion)`.

**Feature Engineering:** 15+ features including category demand, freight ratios, seller ratings, product dimensions, geolocation-based shipping estimates.

---

## 📝 License

MIT
