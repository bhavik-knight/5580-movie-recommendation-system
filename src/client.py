"""
Movie Recommendation Engine Client
==================================

What the engine does:
---------------------
This engine provides personalized movie recommendations based on a set of input movies provided by the user. 
It analyzes user-item interactions to find similar movies and explains the rationale behind each recommendation.

What it doesn't do:
-------------------
- No real-time data: The engine uses a static snapshot of the MovieLens 100K dataset.
- No user history: Recommendations are based solely on the movies provided in the current session, not on past ratings.
- Based on 1997 data: All movies and ratings reflect the state of the cinema and user preferences from the late 90s.

Algorithm used:
---------------
Item-Item Collaborative Filtering via Cosine Similarity.
The engine calculates similarity scores between movies by comparing their rating vectors across all users. 
A mean-centering normalization is applied to remove individual user rating bias.

Dataset description:
--------------------
- Dataset: MovieLens 100K
- Scale: 943 users, 1682 movies
- Volume: 100,000 ratings

Assumptions:
------------
- Minimum Popularity: Only movies with at least 20 ratings are included for quality.
- Normalization: Mean-centered ratings are used to ensure similarities are captured relative to user averages.
- Explanations: Shared genres are used as a primary explanation for similarity.

Known limitations:
------------------
- Dataset age: Does not include modern films.
- Demographic skew: The original 1997 dataset has a significant male-user bias.
- Sparsity: Only a fraction of the possible user-movie interaction matrix is populated.
"""

import sys
import logging
from pathlib import Path
from src.config import ITEM_SIMILARITY_FILE, MOVIE_LOOKUP_FILE, OUTPUT_DIR
from src.recommender import recommend, load_data
import src.ratings_matrix as ratings_matrix
import src.item_similarity as item_similarity

# Configure error logging
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
error_log = OUTPUT_DIR / "error.log"

def check_pipeline():
    """
    Check if precomputed files exist; if not, run the pipeline.
    """
    if not (ITEM_SIMILARITY_FILE.exists() and MOVIE_LOOKUP_FILE.exists()):
        print("\nPrecomputed data not found. Running full pipeline (this may take a few seconds)...")
        try:
            ratings_matrix.main()
            item_similarity.main()
            print("Pipeline completed successfully.\n")
        except Exception as e:
            with open(error_log, "a") as f:
                f.write(f"Pipeline Error: {str(e)}\n")
            print("Critical error during data precomputation. Check output/error.log.")
            sys.exit(1)
    else:
        print("Precomputed data found. Loading fast path...")

def get_user_inputs(title_to_id):
    """
    Interactively collect up to 5 valid movie titles from the user.
    """
    inputs = []
    print("\nPlease enter up to 5 movies you like (Titles must match 1997 MovieLens database).")
    
    while len(inputs) < 5:
        movie = input(f"Movie {len(inputs) + 1}: ").strip()
        
        if movie in title_to_id:
            inputs.append(movie)
            if len(inputs) < 5:
                another = input("Add another movie? (y/n): ").lower().strip()
                if another != 'y':
                    break
        else:
            print(f"Warning: Movie '{movie}' not found in our database. Please try another title.")
            print("Tip: Make sure to include the year, e.g., 'Star Wars (1977)'")
            
    return inputs

def run_cli():
    """
    Runs the interactive recommendation engine CLI.
    """
    print("============================================")
    print("Welcome to the MovieLens Recommendation Engine!")
    print("Tell us what you like, and we'll find related classics.")
    print("============================================")

    check_pipeline()

    # Load data once to get the title lookup
    try:
        _, title_to_id, _, _ = load_data()
    except Exception as e:
        with open(error_log, "a") as f:
            f.write(f"Data Loading Error: {str(e)}\n")
        print("Error loading movie titles. Please verify your data directory.")
        return

    while True:
        user_movies = get_user_inputs(title_to_id)
        
        if not user_movies:
            print("\nNo valid movies entered. We need at least one to start.")
        else:
            print("\nCalculating recommendations for your tastes...")
            try:
                results = recommend(user_movies, top_n=10)
                
                if results:
                    print("\n=== TOP RECOMMENDATIONS ===")
                    for rec in results:
                        print(f"{rec['rank']}. {rec['title']} [Score: {rec['score']:.4f}]")
                        print(f"   Why: {rec['reason']}")
                else:
                    print("\nCould not find strong matches for that combination. Try adding more variety!")
            except Exception as e:
                with open(error_log, "a") as f:
                    f.write(f"Recommendation Error: {str(e)}\n")
                print("\nAn error occurred while generating recommendations. Logged to output/error.log.")

        retry = input("\nWould you like to try again with different movies? (y/n): ").lower().strip()
        if retry != 'y':
            break

    print("\nThank you for using the MovieLens Recommendation Engine. Goodbye!")

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\n\nSession ended by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        with open(error_log, "a") as f:
            f.write(f"Unexpected Crash: {str(e)}\n")
        print("\nAn unexpected error occurred. Details saved to output/error.log.")
        sys.exit(1)
