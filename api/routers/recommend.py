"""
FastAPI router for movie recommendation endpoints.
Handles health checks, movie discovery, and recommendation generation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import api.dependencies as deps
from api.models import (
    RecommendRequest, RecommendResponse,
    HealthResponse, MovieDetailResponse, MoviesListResponse
)
from api.config import get_settings, Settings
from api.services import RecommenderService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Check the health and status of the API and recommender service.
    
    Returns:
        HealthResponse: API status, version, and data loading state.
    """
    try:
        service_ready = deps._recommender_service is not None and deps._recommender_service.is_loaded
        status = "ok" if service_ready else "loading"
        
        movies_loaded = 0
        if deps._recommender_service and deps._recommender_service.title_lookup:
            movies_loaded = len(deps._recommender_service.title_lookup)
            
        return HealthResponse(
            status=status,
            app_name=settings.app_name,
            app_version=settings.app_version,
            movies_loaded=movies_loaded
        )
    except Exception as e:
        logger.error(f"Health check endpoint failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/movies", response_model=MoviesListResponse)
async def list_movies(
    search: Optional[str] = Query(None, description="Filter movies by title substring (case-insensitive)"),
    service: RecommenderService = Depends(deps.get_recommender_service)
):
    """
    Get a sorted list of all available movie titles with optional search filtering.
    
    Args:
        search (str, optional): Substring to search for in titles.
        service (RecommenderService): The loaded recommender service.
        
    Returns:
        MoviesListResponse: Total count and list of matching movie titles.
    """
    try:
        all_movies = service.get_movies_list()
        
        if search:
            search_lower = search.lower()
            filtered_movies = [title for title in all_movies if search_lower in title.lower()]
        else:
            filtered_movies = all_movies
            
        return MoviesListResponse(
            total=len(filtered_movies),
            movies=filtered_movies
        )
    except Exception as e:
        logger.error(f"Movies list endpoint failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(
    request: RecommendRequest,
    service: RecommenderService = Depends(deps.get_recommender_service)
):
    """
    Generate movie recommendations based on a list of input titles.
    
    Args:
        request (RecommendRequest): The user's movie preferences.
        service (RecommenderService): The loaded recommender service.
        
    Returns:
        RecommendResponse: Top recommendations with scores and reasoning.
    """
    try:
        result = service.recommend(request.titles, request.top_n)
        return RecommendResponse(**result)
    except ValueError as ve:
        logger.error(f"Validation error in recommendation request: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Recommendation generation failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while generating recommendations.")

@router.get("/movie/{movie_id}", response_model=MovieDetailResponse)
async def get_movie_details(
    movie_id: int,
    service: RecommenderService = Depends(deps.get_recommender_service)
):
    """
    Get detailed information and ratings for a specific movie.
    
    Args:
        movie_id (int): Unique ID of the movie.
        service (RecommenderService): The loaded recommender service.
        
    Returns:
        MovieDetailResponse: Detailed metadata and ratings for the movie.
    """
    try:
        movie_detail = service.get_movie_detail(movie_id)
        return MovieDetailResponse(**movie_detail)
    except ValueError as ve:
        logger.error(f"Movie detail requested for invalid ID {movie_id}: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Movie detail endpoint failure for ID {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
