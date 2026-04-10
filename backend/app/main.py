import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import LATEST_PERIOD
from app.data_loader import build_dataset
from app.routers import area, forecast, providers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading dataset...")
    app.state.dataset = build_dataset()
    logger.info("Dataset loaded: %d rows", len(app.state.dataset))
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Broadband Market Scanner API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response time monitoring middleware ---
@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    path = request.url.path
    if path.startswith("/api/"):
        level = logging.WARNING if elapsed_ms > 2000 else logging.INFO
        logger.log(
            level,
            "%s %s -> %d (%.0fms)%s",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
            " [SLOW]" if elapsed_ms > 2000 else "",
        )

    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.0f}"
    return response


app.include_router(area.router, prefix="/api/v1")
app.include_router(providers.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    row_count = len(app.state.dataset) if hasattr(app.state, "dataset") else 0
    return {
        "status": "ok",
        "rows_loaded": row_count,
        "data_as_of": LATEST_PERIOD,
    }
