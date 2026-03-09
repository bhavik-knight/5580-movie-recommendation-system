import logging
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application-wide settings for the Movie Recommendation API.
    Values can be overridden using environment variables or a .env file.
    """
    app_name: str = "Movie Recommendation API"
    app_version: str = "1.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]
    min_rating_threshold: int = 20
    default_top_n: int = 10
    max_input_movies: int = 5
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    log_file: Path = Path("output/api.log")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def configure_logging(settings: Settings) -> None:
    """
    Configure global logging settings for the application.
    
    Args:
        settings (Settings): Global application settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_file, mode="a")
        ]
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    """
    return Settings()
