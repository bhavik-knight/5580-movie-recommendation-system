"""
Core Recommendation Engine module.
Implements item-item collaborative filtering using cosine similarity.
Given a set of input movies, it generates ranked recommendations with explanations.
"""

import pandas as pd
import logging
from datetime import datetime
from src.config import (
    U_ITEM_PATH, U_ITEM_NAMES, GENRE_COLUMNS,
    ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE
)

def load_data() -> tuple[pd.DataFrame, dict[str, int], dict[int, str], pd.DataFrame]:
    """
    Load precomputed similarity matrix and metadata for the recommender.
    
    Returns:
        tuple: (similarity_df, title_to_id, id_to_title, u_item)
    """
    # Load item-item similarity matrix (Movies x Movies)
    similarity_df = pd.read_csv(ITEM_SIMILARITY_FILE, index_col=0)
    similarity_df.columns = similarity_df.columns.astype(int)
    
    # Load movie title lookup table
    lookup_df = pd.read_csv(MOVIE_LOOKUP_FILE)
    title_to_id = dict(zip(lookup_df['title'], lookup_df['movie_id']))
    id_to_title = dict(zip(lookup_df['movie_id'], lookup_df['title']))
    
    # Load raw item data for genre retrieval
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")
    
    return similarity_df, title_to_id, id_to_title, u_item

def recommend(input_titles: list[str], top_n: int = 10, min_score: float = 0.05) -> list[dict]:
    """
    Generate movie recommendations based on a list of input titles.
    
    Args:
        input_titles (list[str]): List of 1 to 5 movie titles provided by the user.
        top_n (int, optional): Maximum number of recommendations to return (capped at 10). Defaults to 10.
        min_score (float, optional): Minimum similarity threshold for results. Defaults to 0.05.
        
    Returns:
        list[dict]: Ranked list of recommendations containing metadata and explanation.
        
    Raises:
        ValueError: If input_titles is empty or exceeds the limit of 5 movies.
    """
    # Enforcement of assignment boundaries
    if not input_titles:
        raise ValueError("input_titles list cannot be empty")
        
    if len(input_titles) > 5:
        raise ValueError("Maximum 5 input titles allowed as per assignment requirements")

    # Enforce maximum of 10 recommendations as per assignment requirements
    top_n = min(top_n, 10)
    
    # Load necessary similarity and metadata
    similarity_df, title_to_id, id_to_title, u_item = load_data()
    
    input_ids = []
    valid_titles = []
    
    # Translate titles to IDs and validate existence
    for title in input_titles:
        if title in title_to_id:
            input_ids.append(title_to_id[title])
            valid_titles.append(title)
        else:
            print(f"Warning: Movie '{title}' not found in the dataset. Skipping.")
            
    if not input_ids:
        print("No valid input movies found. Cannot generate recommendations.")
        return []
    
    # Collaborative Filtering Logic:
    # 1. Extract similarity profiles for all input movies
    sim_scores_subset = similarity_df.loc[input_ids]
    
    # 2. Average the scores across the input set
    avg_sim_scores = sim_scores_subset.mean(axis=0)
    
    # 3. Clean results: remove self-recommendations and apply threshold
    avg_sim_scores = avg_sim_scores.drop(input_ids)
    avg_sim_scores = avg_sim_scores[avg_sim_scores > min_score]
    
    # 4. Sort and cap results
    avg_sim_scores = avg_sim_scores.sort_values(ascending=False)
    
    if len(avg_sim_scores) == 0:
        print(f"No movies met the minimum similarity threshold of {min_score}.")
        return []
        
    top_scores = avg_sim_scores.head(top_n)
    if len(avg_sim_scores) < top_n:
        print(f"Note: Only {len(avg_sim_scores)} movies met the threshold of {min_score}.")
        
    # Build enriched recommendation objects with explanations
    recommendations = []
    input_u_item = u_item[u_item['movie_id'].isin(input_ids)]
    
    for rank, (movie_id, score) in enumerate(top_scores.items(), start=1):
        movie_row = u_item[u_item['movie_id'] == movie_id]
        if movie_row.empty:
            continue
            
        movie_title = id_to_title.get(movie_id, "Unknown Title")
        
        # Collaborative Explanation logic: check for shared genres first
        shared_genres = set()
        for genre in GENRE_COLUMNS:
            if movie_row[genre].values[0] == 1 and input_u_item[genre].sum() > 0:
                shared_genres.add(genre)
                
        if shared_genres:
            reason = f"Shares genres with your input: {', '.join(sorted(list(shared_genres)))}"
        else:
            reason = "Highly rated by users who liked your input movies"
            
        recommendations.append({
            "rank": rank,
            "title": movie_title,
            "score": score,
            "reason": reason
        })
        
    return recommendations

def print_recommendations(results: list[dict]) -> None:
    """
    Format and print a list of recommendation dictionaries to the console.
    
    Args:
        results (list[dict]): Recommendations produced by recommend().
    """
    if not results:
        print("No recommendations to display.")
        return
        
    print("\n" + "="*30)
    print("      TOP RECOMMENDATIONS")
    print("="*30)
    for rec in results:
        print(f"{rec['rank']}. {rec['title']}  [Score: {rec['score']:.4f}]")
        print(f"   Why: {rec['reason']}")
    print("="*30 + "\n")

def main() -> None:
    """
    Main execution flow: runs predefined evaluation cases and logs outcomes.
    """
    # Log to both terminal and file for evaluation records
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("output/recommendation_test_results.txt", mode="w")
        ]
    )

    logging.info(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*40)
    logging.info("   RECOMMENDER SYSTEM EVALUATION")
    logging.info("="*40 + "\n")
    
    test_cases = [
        ["Star Wars (1977)"],
        ["Toy Story (1995)", "Aladdin (1992)"],
        ["Fargo (1996)", "Pulp Fiction (1994)", "Silence of the Lambs, The (1991)"]
    ]
    
    for case in test_cases:
        logging.info(f"--- Input: {case} ---")
        try:
            results = recommend(case, top_n=10)
            
            if results:
                logging.info("RECOMMENDATIONS:")
                for rec in results:
                    logging.info(f"{rec['rank']}. {rec['title']}  [Score: {rec['score']:.4f}]")
                    logging.info(f"   Why: {rec['reason']}")
            else:
                logging.info("No recommendations found.")
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            
        logging.info("-" * 20 + "\n")

    print(f"Evaluation results saved to: output/recommendation_test_results.txt")

if __name__ == "__main__":
    main()
