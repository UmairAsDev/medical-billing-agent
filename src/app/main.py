from fastapi import FastAPI
from src.app.api import router
from src.core.config import settings
from loguru import logger

def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME)
    
    # Health Check (Critical for AWS Load Balancers)
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "env": settings.ENV}

    app.include_router(router, prefix="/api/v1")
    
    logger.info("Application Startup Complete")
    return app

app = create_app()