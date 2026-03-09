"""
Business logic layer for the Movie Recommendation API.
This service wraps the core recommendation logic and manages data loading.
"""

import logging
from pathlib import Path
import pandas as pd
from api.config import Settings
from src.recommender import recommend as core_recommend

logger = logging.getLogger(__name__)

class RecommenderService:
    """
    Service responsible for loading recommendation data and providing 
    recommendation, searching, and metadata retrieval services.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the recommender service.
        
        Args:
            settings (Settings): Application settings containing paths and constants.
        """
        self.settings = settings
        self.similarity_matrix: pd.DataFrame | None = None
        self.title_lookup: dict[str, int] | None = None
        self.id_to_title: dict[int, str] | None = None
        self.genre_data: pd.DataFrame | None = None
        self.avg_ratings: pd.Series | None = None
        self.total_ratings: pd.Series | None = None
        self.movie_count: int = 0
        self.is_loaded: bool = False

    def load(self) -> None:
        """
        Load all required data files from the data and output directories into memory.
        
        Raises:
            RuntimeError: If any required files are missing or fail to load.
        """
        try:
            # 1. Load similarity matrix
            sim_path = self.settings.output_dir / "item_similarity_matrix.csv"
            if not sim_path.exists():
                raise FileNotFoundError(f"Similarity matrix not found at {sim_path}")
            
            self.similarity_matrix = pd.read_csv(sim_path, index_col=0)
            self.similarity_matrix.columns = self.similarity_matrix.columns.astype(int)
            logger.info("Successfully loaded similarity matrix.")

            # 2. Load title lookup
            lookup_path = self.settings.output_dir / "movie_id_title_lookup.csv"
            if not lookup_path.exists():
                raise FileNotFoundError(f"Movie lookup file not found at {lookup_path}")
            
            lookup_df = pd.read_csv(lookup_path)
            self.movie_count = len(lookup_df)
            self.title_lookup = dict(zip(lookup_df['title'], lookup_df['movie_id']))
            self.id_to_title = dict(zip(lookup_df['movie_id'], lookup_df['title']))
            logger.info("Successfully loaded movie title lookups.")

            # 3. Load genre data from u.item
            item_path = self.settings.data_dir / "u.item"
            if not item_path.exists():
                raise FileNotFoundError(f"Movie data not found at {item_path}")
            
            from src.config import U_ITEM_NAMES, GENRE_COLUMNS
            full_item_data = pd.read_csv(item_path, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
            # Keep only movie_id and genre columns
            self.genre_data = full_item_data[['movie_id'] + GENRE_COLUMNS]
            logger.info("Successfully loaded genre data.")

            # 4. Load average ratings from ratings_matrix.csv
            ratings_path = self.settings.output_dir / "ratings_matrix.csv"
            if not ratings_path.exists():
                logger.warning(f"Ratings matrix not found at {ratings_path}. Average ratings will be unavailable.")
                self.avg_ratings = pd.Series()
                self.total_ratings = pd.Series()
            else:
                raw_ratings = pd.read_csv(ratings_path, index_col=0)
                self.avg_ratings = raw_ratings.mean(axis=0)
                self.total_ratings = raw_ratings.count(axis=0)
                # Convert index to int since it comes back as string from CSV
                self.avg_ratings.index = self.avg_ratings.index.astype(int)
                self.total_ratings.index = self.total_ratings.index.astype(int)
                logger.info("Successfully calculated average and total ratings.")

            self.is_loaded = True
            logger.info("RecommenderService initialization complete.")

        except Exception as e:
            logger.error(f"Failed to load RecommenderService data: {str(e)}")
            raise RuntimeError(f"Service startup failed: {str(e)}") from e

    def get_movies_list(self) -> list[str]:
        """
        Return a sorted list of all available movie titles.
        
        Returns:
            list[str]: Alphabetically sorted movie titles.
            
        Raises:
            RuntimeError: If the service has not been loaded.
        """
        if not self.is_loaded:
            logger.error("Attempted to access movies list before loading service.")
            raise RuntimeError("Service not loaded.")
            
        return sorted(list(self.title_lookup.keys()))

    def get_movie_detail(self, movie_id: int) -> dict:
        """
        Retrieve metadata and rating statistics for a specific movie.
        
        Args:
            movie_id (int): The unique ID of the movie.
            
        Returns:
            dict: Movie details including title, genres, average rating, and total ratings.
            
        Raises:
            ValueError: If the movie_id is not found in the dataset.
            RuntimeError: If the service has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Service not loaded.")

        if movie_id not in self.id_to_title:
            logger.error(f"Movie detail requested for unknown ID: {movie_id}")
            raise ValueError(f"Movie with ID {movie_id} not found.")

        title = self.id_to_title[movie_id]
        
        # Get genres
        from src.config import GENRE_COLUMNS
        movie_genre_row = self.genre_data[self.genre_data['movie_id'] == movie_id]
        genres = [g for g in GENRE_COLUMNS if movie_genre_row[g].values[0] == 1]
        
        # Get ratings
        avg_rating = self.avg_ratings.get(movie_id, 0.0)
        total_counts = self.total_ratings.get(movie_id, 0)

        return {
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            "average_rating": float(avg_rating),
            "total_ratings": int(total_counts)
        }

    def recommend(self, titles: list[str], top_n: int) -> dict:
        """
        Generate movie recommendations based on user input titles.
        
        Args:
            titles (list[str]): List of movie titles provided by the user.
            top_n (int): Number of recommendations to return.
            
        Returns:
            dict: Recommendation results containing valid titles and the recommendation list.
            
        Raises:
            ValueError: If no valid titles are found in the input.
            RuntimeError: If the service has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Service not loaded.")

        valid_titles = [t for t in titles if t in self.title_lookup]
        invalid_titles = [t for t in titles if t not in self.title_lookup]
        
        if not valid_titles:
            logger.error(f"No valid titles found in recommendation request: {titles}")
            raise ValueError("None of the provided titles were found in the database.")

        try:
            # Call core recommendation logic from src
            results = core_recommend(valid_titles, top_n=top_n)
            
            message = "Success"
            if invalid_titles:
                message = f"Recommendations generated, but some titles were skipped: {', '.join(invalid_titles)}"

            return {
                "input_titles": titles,
                "valid_titles": valid_titles,
                "recommendations": results,
                "message": message
            }
        except Exception as e:
            logger.error(f"Error during recommendation calculation: {str(e)}")
            raise RuntimeError(f"Recommendation generation failed: {str(e)}") from e
