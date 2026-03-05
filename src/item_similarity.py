import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    U_ITEM_PATH, U_ITEM_NAMES, RATINGS_MATRIX_NORM_FILE,
    FILTERED_MOVIE_IDS_FILE, ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE
)

def load_inputs():
    """
    Load the normalized ratings matrix, filtered movie IDs, and movie item data.
    
    Returns:
        tuple: (normalized_matrix, filtered_ids, u_item)
    """
    # Load normalized ratings matrix (index=user_id, columns=movie_id)
    # The first column in the CSV is the user_id index
    normalized_matrix = pd.read_csv(RATINGS_MATRIX_NORM_FILE, index_col=0)
    
    # Ensure column names are integers, since reading from CSV makes them strings
    normalized_matrix.columns = normalized_matrix.columns.astype(int)
    
    # Load filtered movie IDs
    filtered_ids_df = pd.read_csv(FILTERED_MOVIE_IDS_FILE)
    filtered_ids = filtered_ids_df['movie_id'].values
    
    # Load u.item for titles
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
    
    return normalized_matrix, filtered_ids, u_item

def compute_similarity(normalized_matrix):
    """
    Compute item-item cosine similarity on the transposed normalized matrix.
    
    Args:
        normalized_matrix (pd.DataFrame): The mean-centered user-movie ratings matrix.
        
    Returns:
        pd.DataFrame: Item-item similarity matrix.
    """
    # Transpose so movies are rows and users are columns
    item_user_matrix = normalized_matrix.T
    
    # Fill remaining NaNs with 0 in the transposed matrix (movies unfound by users)
    # This ensures we don't modify the original normalized_matrix
    item_user_matrix_filled = item_user_matrix.fillna(0)
    
    # Compute cosine similarity
    similarity_array = cosine_similarity(item_user_matrix_filled)
    
    # Wrap in a DataFrame with movie IDs
    movie_ids = item_user_matrix.index
    similarity_df = pd.DataFrame(
        similarity_array,
        index=movie_ids,
        columns=movie_ids
    )
    
    return similarity_df

def create_title_lookup(u_item, filtered_ids):
    """
    Create a movie ID to title lookup dictionary (as a DataFrame).
    
    Args:
        u_item (pd.DataFrame): The movies dataframe.
        filtered_ids (list or np.ndarray): Filtered movie IDs.
        
    Returns:
        pd.DataFrame: Lookup dataframe containing movie_id and title.
    """
    lookup_df = u_item[u_item['movie_id'].isin(filtered_ids)][['movie_id', 'title']]
    return lookup_df

def run_sanity_checks(similarity_df, lookup_df):
    """
    Run sanity checks on the similarity matrix.
    
    Args:
        similarity_df (pd.DataFrame): Item-item similarity matrix.
        lookup_df (pd.DataFrame): Lookup dataframe for movie titles.
    """
    print("\n=== SANITY CHECKS ===")
    
    # Check diagonal
    diagonal = np.diag(similarity_df.values)
    is_diag_one = np.allclose(diagonal, 1.0)
    print(f"Diagonal is all 1.0 (self-similarity): {is_diag_one}")
    
    title_to_id = dict(zip(lookup_df['title'], lookup_df['movie_id']))
    id_to_title = dict(zip(lookup_df['movie_id'], lookup_df['title']))
    
    def get_top_similar(movie_title, top_n=5):
        if movie_title not in title_to_id:
            return f"Movie '{movie_title}' not found in filtered list."
            
        movie_id = title_to_id[movie_title]
        # Get similarities, drop the diagonal (self), sort descending
        sim_scores = similarity_df[movie_id].drop(movie_id)
        top_ids = sim_scores.sort_values(ascending=False).head(top_n).index
        
        results = [f"{id_to_title[m_id]} (Score: {sim_scores[m_id]:.4f})" for m_id in top_ids]
        return "\n  ".join(results)
    
    print("\nTop 5 movies similar to 'Star Wars (1977)':")
    print("  " + get_top_similar("Star Wars (1977)"))
    
    print("\nTop 5 movies similar to 'Toy Story (1995)':")
    print("  " + get_top_similar("Toy Story (1995)"))

def main():
    """
    Orchestrates the item similarity calculation.
    
    Returns:
        tuple: (similarity_matrix, title_lookup_df)
    """
    print("=== ITEM SIMILARITY COMPUTATION ===")
    
    # 1. Load data
    print("\n[Step 1] Loading normalized matrix and movie list...")
    normalized_matrix, filtered_ids, u_item = load_inputs()
    
    # 2. Compute similarity
    print("\n[Step 2] Computing cosine similarity...")
    similarity_df = compute_similarity(normalized_matrix)
    print(f"Similarity matrix shape: {similarity_df.shape} (Movies x Movies)")
    
    # 3. Create title lookup
    print("\n[Step 3] Creating movie title lookup...")
    lookup_df = create_title_lookup(u_item, filtered_ids)
    
    # 4. Sanity checks
    run_sanity_checks(similarity_df, lookup_df)
    
    # 5. Save outputs
    print("\n[Step 4] Saving results...")
    similarity_df.to_csv(ITEM_SIMILARITY_FILE)
    lookup_df.to_csv(MOVIE_LOOKUP_FILE, index=False)
    print(f"Saved similarity matrix to: {ITEM_SIMILARITY_FILE}")
    print(f"Saved title lookup to: {MOVIE_LOOKUP_FILE}")
    
    print("\nDone!")
    return similarity_df, lookup_df

if __name__ == "__main__":
    main()
