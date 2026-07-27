from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.endpoints_ws import router as ws_router
from fastapi.middleware.cors import CORSMiddleware
from core.middlewares import WSConnectionLoggingMiddleware
from core.whisper_ia import get_whisper_service
from core.ai_translator import get_translator_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: pre-load heavy services ──────────────────────────────────
    get_whisper_service()
    get_translator_service()
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────


app = FastAPI(
    title="OBS Translator Backend",
    description="API for real-time speech detection and translation",
    version="1.1.0",
    lifespan=lifespan,
)

# ASGI Middleware for WebSockets (global logging)
app.add_middleware(WSConnectionLoggingMiddleware)

# HTTP CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "OBS Translator Backend is running!"}


app.include_router(ws_router)
