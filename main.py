from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import redis
import sys
from slowapi.errors import RateLimitExceeded

from config import REDIS_URL, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
from api.limiter import limiter
from api.routes.predictions import router as predict_router
from api.services.model_service import model_service
from api.models import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis connection for rate limiting
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_URL}")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    # For tests, we can continue without Redis (rate limiting will be disabled)
    if "pytest" in " ".join(sys.argv):
        logger.warning("Running in test mode - continuing without Redis")
        redis_client = None
    else:
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    logger.info("Starting up...")
    success = model_service.load_model()
    if not success:
        logger.warning(
            "Failed to load model on startup. API will return 503 until model is available."
        )
    yield
    # Shutdown: Cleanup if needed
    logger.info("Shutting down...")
    if redis_client:
        redis_client.close()


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions using logistic regression",
    version="0.1.0",
    lifespan=lifespan,
)

# Set limiter on app state
app.state.limiter = limiter


# Rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        },
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
    )
