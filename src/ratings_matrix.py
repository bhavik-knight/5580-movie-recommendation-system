"""
Ratings Matrix generation module.
Pivots raw rating data into a user-movie matrix, filters for significance,
and performs mean-centering to remove user rating bias.
"""

import pandas as pd
from src.config import (
    U_DATA_PATH, U_ITEM_PATH, U_DATA_NAMES, U_ITEM_NAMES,
    RATINGS_MATRIX_FILE, RATINGS_MATRIX_NORM_FILE, FILTERED_MOVIE_IDS_FILE,
    MATRIX_SUMMARY_FILE
)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw ratings and movie metadata files.
    
    Returns:
        tuple: (u_data, u_item) as pandas DataFrames.
    """
    u_data = pd.read_csv(U_DATA_PATH, sep="\t", names=U_DATA_NAMES)
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
    
    return u_data, u_item

def create_pivot_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-form ratings dataframe into a wide-form user-movie matrix.
    Rows represent user IDs, columns represent movie IDs, and values are ratings.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'user_id', 'movie_id', and 'rating'.
        
    Returns:
        pd.DataFrame: User-movie pivot table with NaNs for unrated movies.
    """
    return df.pivot_table(index='user_id', columns='movie_id', values='rating')

def filter_movies(u_data: pd.DataFrame, threshold: int = 20) -> pd.Index:
    """
    Identify movie IDs that meet or exceed a minimum number of ratings.
    This ensures similarity calculations are based on sufficient sample sizes.
    
    Args:
        u_data (pd.DataFrame): The raw ratings dataframe.
        threshold (int): Minimum number of ratings required to keep a movie.
        
    Returns:
        pd.Index: Subset of unique movie IDs meeting the threshold.
    """
    movie_counts = u_data.groupby('movie_id')['rating'].count()
    return movie_counts[movie_counts >= threshold].index

def normalize_ratings(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform mean-centering per user to remove individual rating bias.
    Subtracts the user's average rating from all of their movie scores.
    
    Args:
        matrix (pd.DataFrame): The pivoted user-movie ratings matrix.
        
    Returns:
        tuple: (normalized_matrix, user_means) where user_means is a Series of averages.
    """
    # Calculate mean rating per user, ignoring NaNs
    user_means = matrix.mean(axis=1)
    
    # Subtract user mean from each row appropriately
    normalized_matrix = matrix.sub(user_means, axis=0)
    
    return normalized_matrix, user_means

def save_outputs(raw_matrix: pd.DataFrame, normalized_matrix: pd.DataFrame, 
                 filtered_ids: pd.Index, summary_text: str) -> None:
    """
    Persist the generated matrices and summary data to the output directory.
    
    Args:
        raw_matrix (pd.DataFrame): The filtered but non-normalized matrix.
        normalized_matrix (pd.DataFrame): The mean-centered matrix.
        filtered_ids (pd.Index): List of movie IDs remaining after threshold filter.
        summary_text (str): Descriptive summary of matrix properties.
    """
    # Save matrices in CSV format
    raw_matrix.to_csv(RATINGS_MATRIX_FILE)
    normalized_matrix.to_csv(RATINGS_MATRIX_NORM_FILE)
    
    # Save the list of IDs used for downstream components
    pd.Series(filtered_ids, name="movie_id").to_csv(FILTERED_MOVIE_IDS_FILE, index=False)
    
    # Save the analytical summary
    with open(MATRIX_SUMMARY_FILE, "w") as f:
        f.write(summary_text)

def main() -> None:
    """
    Main execution flow for building and processing the ratings matrix.
    Handles loading, pivot transformation, filtering, normalization, and persistence.
    """
    print("--- Movie Recommendation System: Ratings Matrix Building ---")
    
    # 1. Loading
    print("Loading datasets...")
    u_data, u_item = load_data()
    
    # 2. Pivoting and Sparsity Analysis
    print("Creating initial user-movie matrix...")
    full_matrix = create_pivot_matrix(u_data)
    
    total_elements = full_matrix.size
    non_nan_elements = full_matrix.count().sum()
    nan_elements = total_elements - non_nan_elements
    sparsity = (nan_elements / total_elements) * 100
    
    print(f"Initial Shape: {full_matrix.shape[0]} Users x {full_matrix.shape[1]} Movies")
    print(f"Matrix Sparsity: {sparsity:.2f}%")
    
    # 3. Filtering
    threshold = 20
    print(f"Filtering movies with fewer than {threshold} ratings...")
    filtered_ids = filter_movies(u_data, threshold=threshold)
    
    # Rebuild matrix using only significant movies
    u_data_filtered = u_data[u_data['movie_id'].isin(filtered_ids)]
    raw_matrix = create_pivot_matrix(u_data_filtered)
    print(f"Movies remaining: {len(filtered_ids)}")
    print(f"Filtered Shape: {raw_matrix.shape}")
    
    # 4. Normalization
    print("Normalizing ratings via per-user mean-centering...")
    normalized_matrix, user_means = normalize_ratings(raw_matrix)
    
    # 5. Summary and Persistence
    print("Processing summary and saving outputs...")
    summary_text = (
        f"=== RATINGS MATRIX SUMMARY ===\n"
        f"Matrix shape: {full_matrix.shape[0]} × {full_matrix.shape[1]}\n"
        f"Sparsity: {sparsity:.2f}% sparse\n"
        f"Movies remaining after threshold filter ({threshold}+ ratings): {len(filtered_ids)} movies\n"
    )
    
    save_outputs(raw_matrix, normalized_matrix, filtered_ids, summary_text)
    
    print(f"Successfully saved matrices and filtered IDs to: {FILTERED_MOVIE_IDS_FILE.parent}")
    print("Done!")

if __name__ == "__main__":
    main()
