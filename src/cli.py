"""
Interactive Command Line Interface (CLI) for the Movie Recommendation Engine.
This module serves as the primary entry point, orchestrating the pipeline 
and providing a user-friendly way to explore recommendations.
"""

import sys
from pathlib import Path
from src.config import ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE, OUTPUT_DIR
from src.recommender import recommend, load_data
import src.ratings_matrix as ratings_matrix
import src.item_similarity as item_similarity

# Persistence path for internal error tracking
ERROR_LOG_PATH = OUTPUT_DIR / "error.log"

def check_pipeline_readiness() -> None:
    """
    Check if the necessary precomputed similarity data exists.
    If files are missing, automatically triggers the full recomputation pipeline.
    """
    if not (ITEM_SIMILARITY_FILE.exists() and MOVIE_LOOKUP_FILE.exists()):
        print("\nPrecomputed similarity data not found. Initializing full pipeline...")
        print("This process takes a few seconds but only needs to run once.\n")
        try:
            # Execute Step 2: Matrix Building
            ratings_matrix.main()
            # Execute Step 3: Similarity Computation
            item_similarity.main()
            print("\nPipeline completed successfully! Data is now cached for fast loading.\n")
        except Exception as e:
            with open(ERROR_LOG_PATH, "a") as f:
                f.write(f"Pipeline Execution Error: {str(e)}\n")
            print("CRITICAL: Pipeline failed to initialize data. See output/error.log for details.")
            sys.exit(1)
    else:
        print("Precomputed data found. Loading fast path...")

def get_interactive_user_movies(title_to_id: dict[str, int]) -> list[str]:
    """
    Collects a list of movie titles from the user via terminal prompts.
    Provides immediate validation feedback and enforces the input limit.
    
    Args:
        title_to_id (dict): A mapping of movie titles to their dataset IDs.
        
    Returns:
        list[str]: A list of validated movie titles.
    """
    user_inputs = []
    print("\nPlease enter up to 5 movies you like.")
    print("Formatting Tip: Use 'Title (Year)' - e.g., 'Star Wars (1977)'")
    
    while len(user_inputs) < 5:
        idx = len(user_inputs) + 1
        movie = input(f"Movie {idx}: ").strip()
        
        if not movie:
            continue
            
        if movie in title_to_id:
            user_inputs.append(movie)
            if len(user_inputs) < 5:
                another = input("Add another movie? (y/n): ").lower().strip()
                if another != 'y':
                    break
        else:
            print(f"  --> Warning: '{movie}' not found.")
            print("      Note: Search is case-sensitive. Verify the year and title format.")
            
    return user_inputs

def main() -> None:
    """
    Orchestrates the interactive CLI loop.
    Handles startup sequence, input collection, recommendation generation, and graceful exit.
    """
    print("=" * 60)
    print("      Welcome to the Movie Recommendation Engine!      ")
    print("=" * 60)
    print("Powered by item-item collaborative filtering on MovieLens 100K.")
    print("Suggests up to 10 movies based on up to 5 titles you provide.\n")

    # Ensure output dir exists before any writes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline Check (Fast vs Slow Path)
    check_pipeline_readiness()

    # 2. Data Loading
    try:
        _, title_to_id, _, _ = load_data()
    except Exception as e:
        with open(ERROR_LOG_PATH, "a") as f:
            f.write(f"Data Loading Error: {str(e)}\n")
        print("Error: Could not load movie lookup tables. Ensure the data directory is intact.")
        return

    # 3. Interactive Loop
    while True:
        user_movies = get_interactive_user_movies(title_to_id)
        
        if not user_movies:
            print("\nNo movies entered. We need a starting point to recommend!")
        else:
            print("\nSearching for your next favorite movies...")
            try:
                results = recommend(user_movies, top_n=10)
                
                if results:
                    print("\n" + "=" * 40)
                    print("         TOP RECOMMENDATIONS")
                    print("=" * 40)
                    for rec in results:
                        print(f"{rec['rank']}. {rec['title']}  [Score: {rec['score']:.4f}]")
                        print(f"   Why: {rec['reason']}")
                    print("=" * 40)
                else:
                    print("\nNo strong matches found. Try entering a different group of movies.")
            except Exception as e:
                with open(ERROR_LOG_PATH, "a") as f:
                    f.write(f"Recommender Execution Error: {str(e)}\n")
                print("\nAn internal error occurred during calculation. See output/error.log.")

        retry = input("\nWould you like to try again with different movies? (y/n): ").lower().strip()
        if retry != 'y':
            break

    print("\nGoodbye! Enjoy your next movie night!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession terminated by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        with open(ERROR_LOG_PATH, "a") as f:
            f.write(f"Unexpected Termination: {str(e)}\n")
        print("\nThe application encountered an unexpected error and closed. Check output/error.log.")
        sys.exit(1)
