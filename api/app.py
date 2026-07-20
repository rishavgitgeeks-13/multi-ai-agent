"""
FastAPI Application
===================

Entry point for the Editorial Intelligence System REST API.

Endpoints
---------
  GET  /api/health                → service health check
  GET  /api/brands                → list of all configured brands
  GET  /  or /ui                  → single-page browser UI (frontend/index.html)
  POST /api/generate/content      → article / blog workflow
  POST /api/generate/email        → email campaign workflow
  POST /api/generate/seo          → SEO-optimised content workflow
  POST /api/generate/social       → LinkedIn / carousel / X workflow

Run
---
    uvicorn api.app:app --reload --port 8000

Or via main.py:
    python main.py
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from api.routes.generate import router as generate_router

from config.settings import settings

# Route modules
from api.routes.content import router as content_router
from api.routes.email import router as email_router
from api.routes.seo import router as seo_router
from api.routes.social import router as social_router
from api.schemas import BrandInfo, BrandsResponse, ErrorResponse, HealthResponse

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_UI_INDEX = _FRONTEND_DIR / "index.html"


# ==========================================================================
# Lifespan — startup / shutdown
# ==========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (%s)", settings.APP_NAME, settings.ENVIRONMENT)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


# ==========================================================================
# App factory
# ==========================================================================


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Multi-agent editorial intelligence system. "
        "Generates SEO-optimised content, email campaigns, and social posts "
        "using a LangGraph pipeline powered by OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------
# CORS — allow browser UI (served /ui or file://) to call the API
# ------------------------------------------------------------------

_origins = ["*"] if settings.ENVIRONMENT == "development" else [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "null",  # file:// pages report Origin: null
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Global exception handler
# ------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Internal server error", "detail": str(exc)},
    )


# ------------------------------------------------------------------
# Register route groups under /api prefix
# ------------------------------------------------------------------

app.include_router(content_router, prefix="/api")
app.include_router(email_router, prefix="/api")
app.include_router(seo_router, prefix="/api")
app.include_router(social_router, prefix="/api")
app.include_router(generate_router, prefix="/api",)


# ==========================================================================
# Utility endpoints
# ==========================================================================


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version="1.0.0",
        environment=settings.ENVIRONMENT,
    )


@app.get(
    "/api/brands",
    response_model=BrandsResponse,
    tags=["System"],
    summary="List all configured brands",
)
async def list_brands() -> BrandsResponse:
    """Load brand configurations from brands.yaml and return the list."""
    try:
        from brands.brand_loader import BrandLoader
        loader = BrandLoader()
        all_brands = loader.get_all_brands()

        brand_list = [
            BrandInfo(
                id=b.get("namespace", b.get("brand", "")),
                display_name=b.get("display_name", ""),
                tone=b.get("tone", ""),
                reader_segment=b.get("reader_segment", []),
                cta=b.get("cta", ""),
                namespace=b.get("namespace", ""),
            )
            for b in all_brands.values()
        ]
    except Exception as exc:
        logger.warning("Could not load brands: %s — returning empty list", exc)
        brand_list = []

    return BrandsResponse(brands=brand_list, total=len(brand_list))


# ==========================================================================
# Browser UI (single HTML page — no Streamlit)
# ==========================================================================


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
async def browser_ui():
    """Serve the single-page HTML frontend."""
    if not _UI_INDEX.exists():
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "frontend/index.html not found"},
        )
    return FileResponse(_UI_INDEX, media_type="text/html; charset=utf-8")


@app.get("/", include_in_schema=False)
async def root():
    """Prefer the browser UI; fall back to a short JSON notice."""
    if _UI_INDEX.exists():
        return FileResponse(_UI_INDEX, media_type="text/html; charset=utf-8")
    return {
        "message": f"{settings.APP_NAME} is running. Visit /docs for the API reference or /ui for the web UI.",
    }

