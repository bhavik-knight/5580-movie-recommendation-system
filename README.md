# Movie Recommendation Engine

A content-aware movie recommendation system built on the MovieLens 100K dataset
using item-item collaborative filtering via cosine similarity.

## Dataset
- Source: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- 943 users, 1682 movies, 100,000 ratings
- Collected September 1997 — April 1998

## Project Structure
(leave this empty for now — we will fill it in at the end)

## Setup
### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Install dependencies
cd into project root then:
```bash
uv sync
```

## Step 1: ETL + Exploratory Data Analysis
Loads and explores all MovieLens data files.

### What it does
- Loads u.data, u.item, u.user, u.genre, u.occupation
- Performs sanity checks on each file
- Analyses rating distribution, top movies, genre distribution, user demographics
- Saves summary to output/eda_summary.txt

### Run it
```bash
python src/etl_eda.py
```

### Expected output
- Prints sanity checks and EDA findings to terminal
- Saves output/eda_summary.txt

## Step 2: Ratings Matrix
Builds the user-movie ratings matrix from raw rating data.

### What it does
- Pivots u.data into a 943 × 1682 user-movie matrix
- Filters out movies with fewer than 20 ratings (939 movies remain)
- Mean-centers ratings per user to remove rating bias
- Saves raw and normalized matrices to output/

### Why mean-centering?
Some users consistently rate high (e.g. always 4-5) while others rate low
(e.g. always 1-3). Mean-centering removes this bias so similarity is based
on relative preferences not absolute scores.

### Assumptions
- Movies with fewer than 20 ratings lack enough signal to recommend reliably
- Missing ratings (movies a user hasn't seen) are left as NaN not 0
- A rating of 0 would imply the user watched and hated it — that's different
  from not having watched it at all

### Run it
```bash
python src/ratings_matrix.py
```

### Expected output
- Matrix shape: 943 × 1682
- Sparsity: ~93.70% sparse
- Movies remaining after threshold filter: 939
- Saves output/ratings_matrix.csv
- Saves output/ratings_matrix_normalized.csv
- Saves output/filtered_movie_ids.csv

## Step 3: Item Similarity Matrix
Computes cosine similarity between all 939 filtered movies.

### What it does
- Loads the normalized ratings matrix from output/
- Transposes it so movies are rows and users are columns
- Fills remaining NaN values with 0 for similarity computation only
- Computes cosine similarity between every pair of movies
- Produces a 939 × 939 similarity matrix
- Saves matrix and movie title lookup to output/

### Why cosine similarity?
Cosine similarity measures the angle between two vectors regardless of
magnitude. Two movies are similar if users who rated one tended to rate
the other similarly — even if one movie has far more ratings than the other.

### Why fill NaN with 0 only for this step?
We only fill NaN with 0 during similarity computation — not in the normalized
matrix itself. A 0 here means "no information" not "the user rated it 0".
This is a deliberate approximation to make the math work.

### Sanity checks performed
- Diagonal of similarity matrix is all 1.0 (a movie is identical to itself)
- Top 5 similar movies to Star Wars (1977) are verified
- Top 5 similar movies to Toy Story (1995) are verified

### Run it
```bash
python src/item_similarity.py
```

### Expected output
- Similarity matrix shape: 939 × 939
- Diagonal check: True
- Saves output/item_similarity_matrix.csv
- Saves output/movie_id_title_lookup.csv

## Step 4: Recommendation Engine
The core recommender — given up to 5 movies, suggests up to 10 more.

### What it does
- Accepts 1 to 5 movie titles as input
- Looks up each title in the similarity matrix
- Averages similarity scores across all input movies
- Excludes input movies from results
- Filters out results below a minimum score threshold of 0.05
- Returns top 10 results ranked by averaged similarity score
- Explains why each movie was recommended

### Algorithm
Item-item collaborative filtering using cosine similarity.
Two movies are considered similar if users who liked one also liked the other.
This is different from content-based filtering which relies purely on genres —
collaborative filtering captures latent patterns in real user behaviour.

### What it does NOT do
- Does not use real-time data — based on 1997/1998 ratings only
- Does not personalise to an individual user's history
- Does not crawl or update automatically
- Does not handle typos or fuzzy title matching — titles must match exactly
- Does not recommend movies outside the 939 filtered movies

### Assumptions
- Minimum similarity score of 0.05 filters out weak recommendations
- Genre overlap is used to explain recommendations where possible
- If no genre overlap exists, explanation defaults to collaborative signal

### Bonus: Why explanations
Every recommendation includes a reason:
- "Shares genres with your input: Action, Thriller" — genre overlap found
- "Highly rated by users who liked your input movies" — pure collaborative signal

### Run it
```bash
python src/recommender.py
```

### Expected output
Runs 3 built-in test cases:
- Single movie input: Star Wars (1977)
- Two movie input: Toy Story (1995), Aladdin (1992)
- Three movie input: Fargo (1996), Pulp Fiction (1994), Silence of the Lambs (1991)
- Saves output/recommendation_test_results.txt

## Step 5: Interactive CLI
The main entry point — runs the full recommendation engine interactively.

### What it does
- Checks if precomputed matrices exist in output/
  - If yes: loads them directly (fast path — skips recomputation)
  - If no: runs the full pipeline automatically then loads results
- Prompts the user to enter up to 5 movie titles one at a time
- Validates each title immediately — warns if not found
- Calls the recommender and displays top 10 results
- Explains why each movie was recommended
- Loops until the user chooses to exit

### Run it
```bash
python src/cli.py
```

### Example session
```
Welcome to the Movie Recommendation Engine!
This engine suggests up to 10 movies based on up to 5 movies you provide.
Powered by item-item collaborative filtering on the MovieLens 100K dataset.

Enter a movie title: Star Wars (1977)
Add another movie? (y/n): n

=== TOP RECOMMENDATIONS ===
1. Return of the Jedi (1983)  [Score: 0.6565]
   Why: Shares genres with your input: Action, Adventure, Romance, Sci_Fi, War
2. Empire Strikes Back, The (1980)  [Score: 0.6320]
   Why: Shares genres with your input: Action, Adventure, Romance, Sci_Fi, War
...

Would you like to try again? (y/n): n
Goodbye! Enjoy your movies!
```

### Notes
- Movie titles must match exactly as they appear in the dataset
- Use the format: Title (Year) e.g. Toy Story (1995)
- Titles are case sensitive
- Run python src/item_similarity.py first if output/ is empty
