"""Standalone Curio inference API, extracted from Curiora Campus."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.routers.curio import curio_router, curio_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await run_in_threadpool(curio_service.release)


app = FastAPI(title="Curio", version="0.1.0", lifespan=lifespan)
app.include_router(curio_router)


@app.get("/health")
def health():
    # Liveness only: model readiness is verified by actual inference.
    return {"status": "ok", "service": "Curio"}
