# SEO Multi-Agent Tool

**Editorial Intelligence System** — a multi-agent content platform that generates brand-aligned, SEO-aware articles, emails, and social posts.

**Purpose:** help readers learn and solve real problems — not chase views, followers, or empty attention. Content should enlighten and fill knowledge gaps (or clearly show the next helpful step without confusion).

It runs a LangGraph pipeline powered by Anthropic Claude (generation + review) and OpenAI embeddings (SEO / vector search), with a FastAPI backend and a Streamlit UI.

---

## Pipeline workflow

All content workflows share the same agent graph:

```
START
  ↓
Manager     → validate input, resolve brand context
  ↓
Research    → KB + web / news / YouTube research package
  ↓
Strategy    → SEO keywords, citations, hashtags, outline
  ↓
Writer      → draft + metadata + formatting (+ citation enrichment)
  ↓
Review      → score draft across 6 dimensions
  ├── score ≥ 95  → PASS → END
  └── score < 95  → FAIL → Writer (rewrite, max 3 cycles)
```

### Agents

| Agent | Role |
|-------|------|
| **Manager** | Validates the request and loads brand config (`brands/brands.yaml`) |
| **Research** | Builds a research package (documents, statistics, citations, sources) |
| **Strategy** | Ranks keywords, formats citations, builds the content outline |
| **Writer** | Produces Markdown; on FAIL, surgically revises the existing draft |
| **Review** | Scores quality and returns rewrite instructions when below target |

### Review scoring (target ≥ 95)

| Dimension | Weight |
|-----------|--------|
| Content quality | 20% |
| SEO compliance | 25% |
| Brand alignment | 20% |
| Structure | 15% |
| Factual grounding | 15% |
| CTA effectiveness | 5% |

If the revision limit is reached before score ≥ 95, the run is **force-passed** with the current score (feedback notes that it is below target).

---

## Content workflows

| Workflow | Output | Typical use |
|----------|--------|-------------|
| **Content** | Article (~2200 words) or blog (~1800 words) | Long-form SEO / authority posts |
| **Email** | Campaign email + subject / preview | Newsletter, nurture, promo, transactional |
| **SEO** | SEO-focused article + keyword / technical analysis | Search-optimised pages |
| **Social** | LinkedIn / carousel / X post + social meta | Short-form social |

API auto-routing (`POST /api/generate`) picks a workflow from intent keywords in the prompt when you do not call a specific endpoint.

---

## Brands

Configured in `brands/brands.yaml`:

| Brand | Focus | Example CTA |
|-------|--------|-------------|
| **GTIB** | M&A advisory | Schedule an Advisory Call |
| **Kinvo** | Premium childcare | Book a Consultation |
| **MPM** | Property | Contact Property Advisor |
| **Futuristix** | Agentic AI / SMBs | Book an AI Discovery Call |
| **GCB** | Engineering services | Contact Our Engineering Team |

Each brand defines tone, audience, pain points, keyword direction, CTA, and (optional) Pinecone namespace for KB retrieval.

---

## Project structure

```
SEO_Multi_Agent_Tool/
├── main.py                 # Starts FastAPI (uvicorn)
├── requirements.txt
├── brands/brands.yaml      # Brand configs
├── agents/                 # LangGraph node functions
├── graphs/                 # StateGraph + review router
├── workflows/              # Content / Email / SEO / Social wrappers
├── api/                    # FastAPI app + routes + schemas
├── frontend/app.py         # Streamlit UI
├── services/               # Research, SEO, writer, review, etc.
├── tools/                  # Tavily, NewsAPI, YouTube
├── memory/                 # MongoDB + Pinecone conversation memory
├── models/                 # Anthropic LLM + OpenAI embeddings
├── prompts/                # Agent prompt contracts (.md)
├── schemas/                # ContentState + Pydantic contracts
├── validators/             # JSON / schema / citation validators
└── config/settings.py      # Env-based settings (pydantic-settings)
```

---

## Prerequisites

- Python **3.10+** (project tested with 3.12)
- API keys listed below (OpenAI **required**; Anthropic **required** for generation)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/gcb-automation/SEO_Multi_Agent_Tool.git
cd SEO_Multi_Agent_Tool

# 2. Create & activate virtualenv (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
# Create a .env file in the project root (see variables below)
```

> Always run with the project `venv`. System Python may miss packages such as `pydantic-settings`.

---

## Environment variables

Create a `.env` in the project root:

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | **Yes** | Embeddings / SEO scoring |
| `ANTHROPIC_API_KEY` | **Yes** (for generation) | Claude writer + review |
| `ANTHROPIC_MODEL` | No (default `claude-sonnet-4-6`) | Model id |
| `TAVILY_API_KEY` | Recommended | Web research |
| `NEWS_API_KEY` | Optional | News research |
| `YOUTUBE_API_KEY` | Optional | YouTube + transcripts |
| `PINECONE_API_KEY` | Optional | Brand knowledge base |
| `PINECONE_INDEX_NAME` | No (default `multi-agent`) | Pinecone index |
| `MONGODB_URI` | Optional | Conversation / run persistence |
| `MONGODB_DATABASE` | No (default `editorial_ai`) | Database name |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT` | Optional | LangSmith tracing |
| `ENVIRONMENT` | No (default `development`) | Enables API reload in development |
| `LOG_LEVEL` | No (default `INFO`) | Logging level |
| `HOST` / `PORT` | No (`0.0.0.0` / `8000`) | API bind address |

Optional Reddit credentials exist in settings but are largely unused (Reddit signals go through Tavily when configured).

---

## How to run

### 1. API (terminal A)

```bash
.\venv\Scripts\Activate.ps1
python main.py
```

- API: http://localhost:8000  
- Swagger: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc  
- Health: http://localhost:8000/api/health  

### 2. Browser UI (recommended — no Streamlit)

Open in your browser:

- http://localhost:8000/  
- or http://localhost:8000/ui  

(or open `frontend/index.html` directly and set the API URL)

### 3. Streamlit UI (optional / legacy)

```bash
.\venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

- UI: http://localhost:8501  
- Point the sidebar API URL at `http://localhost:8000` if needed  

### Example API call

```bash
curl -X POST http://localhost:8000/api/generate/content ^
  -H "Content-Type: application/json" ^
  -d "{\"user_input\":\"How to start startup digital marketing from scratch\",\"content_type\":\"article\",\"brand\":\"Futuristix\",\"objective\":\"seo\",\"language\":\"English\",\"max_revisions\":3}"
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/brands` | List brands from YAML |
| `POST` | `/api/generate` | Auto-route by intent |
| `POST` | `/api/generate/content` | Article / blog |
| `POST` | `/api/generate/email` | Email campaigns |
| `POST` | `/api/generate/seo` | SEO content + analysis |
| `POST` | `/api/generate/social` | LinkedIn / carousel / X |

Common request fields: `user_input`, `brand`, `language`, `additional_instructions`, `session_id`, `max_revisions` (default **3**, max 5 for content).

---

## Key defaults

| Setting | Value |
|---------|-------|
| Review PASS threshold | **95** |
| Max review → writer cycles | **3** |
| Article target length | ~2200 words (min 1200 / max 2500) |
| Blog target length | ~1800 words |

---

## Notes

- Generation can take several minutes (research + write + up to 3 review cycles).
- Streamlit escapes `$` in Markdown so currency like `$500` is not rendered as LaTeX.
- Conversation memory (MongoDB / Pinecone) is optional; failures there are non-fatal.
- Prefer package `pinecone` over legacy `pinecone-client` if you hit Pinecone import warnings.

---

## Documentation

| File | Audience |
|------|----------|
| `Readme.md` | Setup, run commands, API overview |
| `Technical.md` / `Technical.pdf` | Full architecture for engineers |
| `Non-Tech.md` / `Non-Tech.pdf` | Simple English guide for business / marketing / ops |
| `frontend/index.html` | Browser UI (single HTML page — use instead of Streamlit) |

### Browser UI (no Streamlit)

```bash
.\venv\Scripts\Activate.ps1
python main.py
```

Then open: **http://localhost:8000/** or **http://localhost:8000/ui**

When deployed, share only your **domain** (e.g. `https://content.yourcompany.com/ui`) — not the raw server IP. The UI calls `/api/...` on the same host, so users never see an API address in the page.

For local testing with `frontend/index.html` opened as a file, use the **Developer: change API URL** section (defaults to `http://localhost:8000`).

---

## License

Internal use — [gcb-automation/SEO_Multi_Agent_Tool](https://github.com/gcb-automation/SEO_Multi_Agent_Tool).
