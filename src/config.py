from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data File Paths
U_DATA_PATH = DATA_DIR / "u.data"
U_ITEM_PATH = DATA_DIR / "u.item"
U_USER_PATH = DATA_DIR / "u.user"
U_GENRE_PATH = DATA_DIR / "u.genre"
U_OCCUPATION_PATH = DATA_DIR / "u.occupation"

# Output Files
EDA_SUMMARY_FILE = OUTPUT_DIR / "eda_summary.txt"
RATINGS_MATRIX_FILE = OUTPUT_DIR / "ratings_matrix.csv"
RATINGS_MATRIX_NORM_FILE = OUTPUT_DIR / "ratings_matrix_normalized.csv"
FILTERED_MOVIE_IDS_FILE = OUTPUT_DIR / "filtered_movie_ids.csv"
MATRIX_SUMMARY_FILE = OUTPUT_DIR / "matrix_summary.txt"

# Column Definitions
GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", 
    "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", 
    "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"
]

U_DATA_NAMES = ["user_id", "movie_id", "rating", "timestamp"]
U_ITEM_NAMES = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
U_USER_NAMES = ["user_id", "age", "gender", "occupation", "zip_code"]
U_GENRE_NAMES = ["genre_name", "genre_id"]
U_OCCUPATION_NAMES = ["occupation"]

# Plot Filenames
RATING_DIST_PLOT = OUTPUT_DIR / "rating_distribution.png"
GENRE_DIST_PLOT = OUTPUT_DIR / "genre_distribution.png"
USER_AGE_PLOT = OUTPUT_DIR / "user_age_distribution.png"
USER_GENDER_PLOT = OUTPUT_DIR / "user_gender_distribution.png"
