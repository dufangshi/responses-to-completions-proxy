from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.routes.chat_completions import router as chat_completions_router
from app.routes.completions import router as completions_router
from app.services.responses_client import OpenAIResponsesGateway


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    gateway = OpenAIResponsesGateway(settings)
    app.state.settings = settings
    app.state.responses_gateway = gateway
    yield
    await gateway.close()


app = FastAPI(
    title="Completions Compatibility Proxy",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(completions_router)
app.include_router(chat_completions_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
