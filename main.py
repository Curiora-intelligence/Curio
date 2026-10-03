from dotenv import load_dotenv
load_dotenv()

import asyncio
import importlib.util
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from app.routers.curio import curio_router, curio_service
from app.routers.memory import memory_router
from app.routers.discovery import discovery_router
from app.routers.live import live_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await curio_service.close()


app = FastAPI(title="Curio", version="0.2.0", lifespan=lifespan)
app.include_router(curio_router)
app.include_router(memory_router)
app.include_router(discovery_router)
app.include_router(live_router)


@app.exception_handler(SQLAlchemyError)
async def database_error(request: Request, exc: SQLAlchemyError):
    return JSONResponse(status_code=503, content={"detail": "Persistent storage unavailable. Please retry."})


@app.get("/health")
async def health():
    postgres, redis = await asyncio.gather(curio_service.database.health(), curio_service.cache.health())
    runtime = curio_service.gateway._runtime
    return {"status": "ok" if postgres == "ok" else "degraded", "service": "Curio", "postgres": postgres,
            "redis": redis, "models/runtime": {"status": "loaded" if runtime else "not_loaded",
            "runtime": runtime.name if runtime else None,
            "mlx_installed": importlib.util.find_spec("mlx") is not None,
            "torch_installed": importlib.util.find_spec("torch") is not None,
            "resident_mode": getattr(runtime, "_mode", None)}}


@app.get("/live", include_in_schema=False)
async def live_demo():
    return FileResponse(Path(__file__).parent / "static" / "live.html")
