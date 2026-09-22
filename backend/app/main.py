from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.retrieval import router as retrieval_router
from app.api.chat import router as chat_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create local runtime directories before the API accepts requests."""
    for directory_name in ("uploads", "extracted", "chroma", "exports"):
        (settings.data_dir / directory_name).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Local Advanced RAG learning assistant API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(retrieval_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
