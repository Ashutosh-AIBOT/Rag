from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.lifespan import lifespan
from app.routers import health, documents, query, evaluation

# 1. Initialize logging
setup_logging()
logger = get_logger(__name__)

# 2. Initialize FastAPI app with custom lifespan hook
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# 3. Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Include routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(query.router, prefix=settings.API_PREFIX)
app.include_router(query.common_router, prefix=settings.API_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_PREFIX)
app.include_router(evaluation.legacy_router, prefix=settings.API_PREFIX)

# 5. Simple Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc} on URL: {request.url}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred on the server.",
            "detail": str(exc)
        }
    )

# 6. Welcome Root Endpoint
@app.get("/")
async def root():
    logger.info("Root landing page hit")
    return {
        "message": "Welcome to the Advanced RAG Platform API",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online"
    }
