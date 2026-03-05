"""
Item Similarity Matrix computation module.
Calculates cosine similarity between movies based on the normalized user-movie matrix,
enabling content-aware collaborative filtering recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    U_ITEM_PATH, U_ITEM_NAMES, RATINGS_MATRIX_NORM_FILE,
    FILTERED_MOVIE_IDS_FILE, ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE
)

def load_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load necessary inputs: normalized ratings matrix, filtered movie IDs, and full movie metadata.
    
    Returns:
        tuple: (normalized_matrix, filtered_ids, u_item) as pandas DataFrames and numpy array.
    """
    # Load normalized ratings matrix (index=user_id, columns=movie_id)
    normalized_matrix = pd.read_csv(RATINGS_MATRIX_NORM_FILE, index_col=0)
    
    # Ensure column names are integers after CSV reload
    normalized_matrix.columns = normalized_matrix.columns.astype(int)
    
    # Load list of filtered movie IDs
    filtered_ids_df = pd.read_csv(FILTERED_MOVIE_IDS_FILE)
    filtered_ids = filtered_ids_df['movie_id'].values
    
    # Load raw item data for title resolution
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
    
    return normalized_matrix, filtered_ids, u_item

def compute_similarity(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item-item cosine similarity on the normalized user-movie ratings.
    Transposes the matrix to treat movies as vectors across the user feature space.
    
    Args:
        normalized_matrix (pd.DataFrame): The mean-centered user-movie ratings matrix.
        
    Returns:
        pd.DataFrame: A symmetrical Item-Item similarity matrix with movie IDs as index/columns.
    """
    # Transpose so movies are rows (vectors to compare)
    item_user_matrix = normalized_matrix.T
    
    # Fill remaining NaNs with 0 to enable numerical computation
    # A 0 in this context signifies no information (neutral impact on similarity)
    item_user_matrix_filled = item_user_matrix.fillna(0)
    
    # Calculate cosine similarity matrix
    similarity_array = cosine_similarity(item_user_matrix_filled)
    
    # Wrap result in a DataFrame for labeled access
    movie_ids = item_user_matrix.index
    similarity_df = pd.DataFrame(
        similarity_array,
        index=movie_ids,
        columns=movie_ids
    )
    
    return similarity_df

def create_title_lookup(u_item: pd.DataFrame, filtered_ids: np.ndarray) -> pd.DataFrame:
    """
    Create a mapping between movie IDs and their titles for only the filtered subset of movies.
    
    Args:
        u_item (pd.DataFrame): The raw movie metadata dataframe.
        filtered_ids (np.ndarray): Array of movie IDs meeting the rating threshold.
        
    Returns:
        pd.DataFrame: Mapping dataframe containing ['movie_id', 'title'].
    """
    lookup_df = u_item[u_item['movie_id'].isin(filtered_ids)][['movie_id', 'title']]
    return lookup_df

def run_sanity_checks(similarity_df: pd.DataFrame, lookup_df: pd.DataFrame) -> None:
    """
    Validate the correctness of the similarity matrix through diagonal checks
    and qualitative top-result verification.
    
    Args:
        similarity_df (pd.DataFrame): The computed item similarity matrix.
        lookup_df (pd.DataFrame): ID-to-title mapping dataframe.
    """
    print("--- Similarity Matrix: Sanity Checks ---")
    
    # 1. Identity Verification: Movie identity should result in similarity of 1.0
    diagonal = np.diag(similarity_df.values)
    is_diag_one = np.allclose(diagonal, 1.0)
    print(f"Self-similarity Check (Diagonal equals 1.0): {is_diag_one}")
    
    # Setup mapping for verification
    title_to_id = dict(zip(lookup_df['title'], lookup_df['movie_id']))
    id_to_title = dict(zip(lookup_df['movie_id'], lookup_df['title']))
    
    def get_top_similar(movie_title: str, top_n: int = 5) -> str:
        if movie_title not in title_to_id:
            return f"Movie '{movie_title}' not found in filtered list."
            
        movie_id = title_to_id[movie_title]
        # Drop self-similarity and sort by score
        sim_scores = similarity_df[movie_id].drop(movie_id)
        top_ids = sim_scores.sort_values(ascending=False).head(top_n).index
        
        results = [f"{id_to_title[m_id]} (Score: {sim_scores[m_id]:.4f})" for m_id in top_ids]
        return "\n  - ".join(results)
    
    # 2. Qualitative validation for known classics
    print("\nTop 5 movies similar to 'Star Wars (1977)':")
    print("  - " + get_top_similar("Star Wars (1977)"))
    
    print("\nTop 5 movies similar to 'Toy Story (1995)':")
    print("  - " + get_top_similar("Toy Story (1995)"))

def main() -> None:
    """
    Main execution flow for computing the item similarity matrix.
    """
    print("--- Movie Recommendation System: Similarity Matrix Computation ---")
    
    # 1. Input Loading
    print("Loading normalized matrix and movie metadata...")
    normalized_matrix, filtered_ids, u_item = load_inputs()
    
    # 2. Computation
    print("Computing cosine similarity matrix across all filtered movies...")
    similarity_df = compute_similarity(normalized_matrix)
    print(f"Computed matrix shape: {similarity_df.shape[0]} x {similarity_df.shape[1]}")
    
    # 3. Lookup Translation
    print("Generating movie title lookup table...")
    lookup_df = create_title_lookup(u_item, filtered_ids)
    
    # 4. Validation
    run_sanity_checks(similarity_df, lookup_df)
    
    # 5. Persistence
    print("\nSaving results to disk...")
    similarity_df.to_csv(ITEM_SIMILARITY_FILE)
    lookup_df.to_csv(MOVIE_LOOKUP_FILE, index=False)
    
    print(f"Item similarity matrix: {ITEM_SIMILARITY_FILE}")
    print(f"Movie title lookup: {MOVIE_LOOKUP_FILE}")
    print("Done!")

if __name__ == "__main__":
    main()
