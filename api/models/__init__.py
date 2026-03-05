"""Model layer for the recommendation API."""

from .recommend import (
    RecommendRequest,
    MovieResult,
    RecommendResponse,
    HealthResponse,
    MovieDetailResponse,
    MoviesListResponse
)

__all__ = [
    "RecommendRequest",
    "MovieResult",
    "RecommendResponse",    "HealthResponse",
    "MovieDetailResponse",
    "MoviesListResponse"
]
