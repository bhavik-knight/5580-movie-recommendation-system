"""
Configuration and fixtures for pytest session.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.recommender import recommend
from src.config import ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE

@pytest.fixture(scope="session")
def similarity_matrix():
    """
    Load similarity matrix once for entire test session.
    
    Returns:
        pd.DataFrame: The item-item similarity matrix.
    """
    if not ITEM_SIMILARITY_FILE.exists():
        pytest.fail(f"Similarity matrix not found at {ITEM_SIMILARITY_FILE}. Run the pipeline first.")
    
    df = pd.read_csv(ITEM_SIMILARITY_FILE, index_col=0)
    df.columns = df.columns.astype(int)
    return df

@pytest.fixture(scope="session")
def title_lookup():
    """
    Load movie title lookup once, return as dict title -> movie_id.
    
    Returns:
        dict: Mapping from movie title to movie ID.
    """
    if not MOVIE_LOOKUP_FILE.exists():
        pytest.fail(f"Movie lookup file not found at {MOVIE_LOOKUP_FILE}. Run the pipeline first.")
    
    df = pd.read_csv(MOVIE_LOOKUP_FILE)
    return dict(zip(df['title'], df['movie_id']))

@pytest.fixture(scope="session")
def valid_single_movie():
    """Returns a list with one valid movie title."""
    return ["Star Wars (1977)"]

@pytest.fixture(scope="session")
def valid_two_movies():
    """Returns a list with two valid movie titles."""
    return ["Toy Story (1995)", "Aladdin (1992)"]

@pytest.fixture(scope="session")
def valid_three_movies():
    """Returns a list with three valid movie titles."""
    return ["Fargo (1996)", "Pulp Fiction (1994)", "Silence of the Lambs, The (1991)"]

@pytest.fixture(scope="session")
def valid_five_movies():
    """Returns a list with five valid movie titles."""
    return [
        "Star Wars (1977)",
        "Fargo (1996)",
        "Toy Story (1995)",
        "Schindler's List (1993)",
        "Pulp Fiction (1994)"
    ]
