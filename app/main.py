"""PLagX – Cross-Language Plagiarism Intelligence. FastAPI application entry."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.papers import router as papers_router
from app.plagiarism import router as plagiarism_router
from plagx.detector import warmup_reference_cache

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
LOG = logging.getLogger(__name__)


app = FastAPI(
    title=get_settings().app_name,
    description="Cross-language plagiarism detection between Dutch and English research papers.",
    version=get_settings().app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware enabled so frontend (separate or same origin) works
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(papers_router)
app.include_router(plagiarism_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def startup_check_environment():
    """Startup hook for logging and environment verification."""
    LOG.info("Frontend API startup complete. Using multilingual detector backend.")
    loaded_docs = warmup_reference_cache()
    LOG.info("Reference cache warmup complete: %d documents ready.", loaded_docs)


@app.get("/")
async def index(request: Request):
    """Serve the dashboard (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "app": get_settings().app_name}
