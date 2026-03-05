import pandas as pd
import numpy as np
from src.config import (
    U_DATA_PATH, U_ITEM_PATH, U_DATA_NAMES, U_ITEM_NAMES,
    RATINGS_MATRIX_FILE, RATINGS_MATRIX_NORM_FILE, FILTERED_MOVIE_IDS_FILE,
    MATRIX_SUMMARY_FILE
)

def load_data():
    """
    Load ratings data and movie item data.
    
    Returns:
        tuple: (u_data, u_item) DataFrames
    """
    u_data = pd.read_csv(U_DATA_PATH, sep="\t", names=U_DATA_NAMES)
    
    # u.item - loading as specified, though we'll primarily use u.data for the matrix
    u_item = pd.read_csv(
        U_ITEM_PATH, 
        sep="|", 
        names=U_ITEM_NAMES, 
        encoding="ISO-8859-1"
    )
    
    return u_data, u_item

def create_pivot_matrix(df):
    """
    Pivot u.data into a user-movie matrix.
    
    Args:
        df (pd.DataFrame): Dataframe containing ratings.
        
    Returns:
        pd.DataFrame: User-movie pivot table.
    """
    # Rows = user_id, Columns = movie_id, Values = rating
    # Missing values remain NaN by default in pivot_table
    matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')
    return matrix

def filter_movies(u_data, threshold=20):
    """
    Filter out movies with fewer than the threshold count of ratings.
    
    Args:
        u_data (pd.DataFrame): The ratings dataframe.
        threshold (int): Minimum number of ratings required.
        
    Returns:
        pd.Index: Indices (movie IDs) of filtered movies.
    """
    movie_counts = u_data.groupby('movie_id')['rating'].count()
    filtered_movie_ids = movie_counts[movie_counts >= threshold].index
    return filtered_movie_ids

def normalize_ratings(matrix):
    """
    Perform mean-centering per user to remove bias.
    
    Args:
        matrix (pd.DataFrame): The user-movie ratings matrix.
        
    Returns:
        tuple: (normalized_matrix, user_means)
    """
    # Calculate mean rating per user (ignoring NaNs)
    user_means = matrix.mean(axis=1)
    
    # Subtract user mean from all their ratings
    normalized_matrix = matrix.sub(user_means, axis=0)
    
    return normalized_matrix, user_means

def save_outputs(raw_matrix, normalized_matrix, filtered_ids, summary_text):
    """
    Save the matrices, filtered IDs, and summary to files.
    
    Args:
        raw_matrix (pd.DataFrame): Original pivoted matrix.
        normalized_matrix (pd.DataFrame): Mean-centered matrix.
        filtered_ids (pd.Index): IDs of movies remaining after filtering.
        summary_text (str): Calculated summary statistics.
    """
    raw_matrix.to_csv(RATINGS_MATRIX_FILE)
    normalized_matrix.to_csv(RATINGS_MATRIX_NORM_FILE)
    pd.Series(filtered_ids, name="movie_id").to_csv(FILTERED_MOVIE_IDS_FILE, index=False)
    
    with open(MATRIX_SUMMARY_FILE, "w") as f:
        f.write(summary_text)

def main():
    """
    Orchestrates the ratings matrix creation process.
    
    Returns:
        tuple: (raw_matrix, normalized_matrix, filtered_movie_ids)
    """
    print("=== BUILDING RATINGS MATRIX ===")
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    u_data, u_item = load_data()
    
    # 2. Build initial matrix
    print("\n[Step 2] Creating initial user-movie matrix...")
    full_matrix = create_pivot_matrix(u_data)
    print(f"Shape: {full_matrix.shape} (Users x Movies)")
    
    # Sparsity calculation
    total_elements = full_matrix.size
    non_nan_elements = full_matrix.count().sum()
    nan_elements = total_elements - non_nan_elements
    sparsity = (nan_elements / total_elements) * 100
    print(f"Sparsity: {sparsity:.2f}% ({nan_elements} NaNs out of {total_elements} cells)")
    
    # 3. Apply threshold filtering
    print("\n[Step 3] Filtering movies with < 20 ratings...")
    filtered_ids = filter_movies(u_data, threshold=20)
    print(f"Movies remaining after filtering: {len(filtered_ids)}")
    
    # Rebuild matrix with filtered movies
    u_data_filtered = u_data[u_data['movie_id'].isin(filtered_ids)]
    raw_matrix = create_pivot_matrix(u_data_filtered)
    print(f"Filtered matrix shape: {raw_matrix.shape}")
    
    # 4. Normalize ratings
    print("\n[Step 4] Normalizing ratings (mean-centering per user)...")
    normalized_matrix, user_means = normalize_ratings(raw_matrix)
    
    print("Per-user mean rating stats:")
    print(f"  Min: {user_means.min():.2f}")
    print(f"  Max: {user_means.max():.2f}")
    print(f"  Overall Mean: {user_means.mean():.2f}")
    
    # 5. Save outputs
    print("\n[Step 5] Saving results...")
    
    summary_text = (
        f"=== RATINGS MATRIX SUMMARY ===\n"
        f"Matrix shape: {full_matrix.shape[0]} × {full_matrix.shape[1]}\n"
        f"Sparsity: {sparsity:.2f}% sparse\n"
        f"Movies remaining after threshold filter: {len(filtered_ids)} movies\n"
    )
    
    save_outputs(raw_matrix, normalized_matrix, filtered_ids, summary_text)
    print(f"Saved raw matrix to: {RATINGS_MATRIX_FILE}")
    print(f"Saved normalized matrix to: {RATINGS_MATRIX_NORM_FILE}")
    print(f"Saved filtered movie IDs to: {FILTERED_MOVIE_IDS_FILE}")
    print(f"Saved matrix summary to: {MATRIX_SUMMARY_FILE}")
    
    print("\nDone!")
    return raw_matrix, normalized_matrix, filtered_ids

if __name__ == "__main__":
    main()
