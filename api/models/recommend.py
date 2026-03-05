"""Pydantic request and response models for the recommendation API."""

from pydantic import BaseModel, validator
from api.config import get_settings

settings = get_settings()

class RecommendRequest(BaseModel):
    """Request model for movie recommendations."""
    titles: list[str]
    top_n: int = 10

    @validator("titles")
    def validate_titles(cls, v: list[str]) -> list[str]:
        if not 1 <= len(v) <= settings.max_input_movies:
            raise ValueError(f"Please provide between 1 and {settings.max_input_movies} movie titles")
        if any(not t.strip() for t in v):
            raise ValueError("Movie titles cannot be empty strings")
        return [t.strip() for t in v]

    @validator("top_n")
    def validate_top_n(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError("top_n must be between 1 and 20")
        return v

class MovieResult(BaseModel):
    """Individual movie recommendation result."""
    rank: int
    title: str
    score: float
    reason: str

class RecommendResponse(BaseModel):
    """Full response for a recommendation request."""
    input_titles: list[str]
    valid_titles: list[str]
    recommendations: list[MovieResult]
    message: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    app_name: str
    app_version: str
    movies_loaded: int

class MovieDetailResponse(BaseModel):
    """Detailed information for a single movie."""
    movie_id: int
    title: str
    genres: list[str]
    average_rating: float
    total_ratings: int

class MoviesListResponse(BaseModel):
    """Response containing a list of all available movies."""
    total: int
    movies: list[str]
