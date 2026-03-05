"""
FastAPI dependencies and app-level state management.
Handles the lifecycle of the RecommenderService and provides it to endpoints.
"""

import asyncio
import logging
from fastapi import HTTPException
from api.services import RecommenderService
from api.config import get_settings

logger = logging.getLogger(__name__)

# Global singleton storage for the service
_recommender_service: RecommenderService | None = None

async def startup_handler() -> None:
    """
    App startup event handler.
    Initializes and loads the RecommenderService in a background thread
    to prevent blocking the main event loop during large file reads.
    
    Raises:
        RuntimeError: If the service fails to load on startup.
    """
    global _recommender_service
    settings = get_settings()
    
    logger.info("Initializing RecommenderService on startup...")
    _recommender_service = RecommenderService(settings)
    
    try:
        # Load large CSV files using to_thread so we don't block other tasks
        await asyncio.to_thread(_recommender_service.load)
        logger.info("RecommenderService loaded successfully.")
    except Exception as e:
        logger.error(f"Critical error during app startup: {str(e)}")
        # We don't exit here so the health check might still function
        # and throw a 503 instead of a crash

async def shutdown_handler() -> None:
    """
    App shutdown event handler.
    Cleans up the RecommenderService instance and releases resources.
    """
    global _recommender_service
    logger.info("Cleaning up RecommenderService on shutdown...")
    _recommender_service = None

def get_recommender_service() -> RecommenderService:
    """
    FastAPI dependency that returns the active RecommenderService instance.
    
    Returns:
        RecommenderService: The loaded and ready-to-use recommendation service.
        
    Raises:
        HTTPException (503): If the service is not yet initialized or failed to load.
    """
    if _recommender_service is None or not _recommender_service.is_loaded:
        logger.warning("Request received while RecommenderService is not available.")
        raise HTTPException(
            status_code=503,
            detail="Recommender service is not ready yet. Please try again shortly."
        )
    return _recommender_service
