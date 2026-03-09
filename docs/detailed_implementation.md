# Detailed Implementation & Technical Notes

This document provides a deep dive into each step of the movie recommendation engine's pipeline, including the algorithms used and the assumptions made during development.

---

## Step 1: ETL + Exploratory Data Analysis
Loads and explores all MovieLens data files.

### What it does
- Loads `u.data`, `u.item`, `u.user`, `u.genre`, `u.occupation`
- Performs sanity checks on each file
- Analyses rating distribution, top movies, genre distribution, user demographics
- Saves summary to `output/eda_summary.txt`

### Run it
```bash
python -m src.etl_eda
```

---

## Step 2: Ratings Matrix
Builds the user-movie ratings matrix from raw rating data.

### What it does
- Pivots `u.data` into a 943 × 1682 user-movie matrix
- Filters out movies with fewer than 20 ratings (939 movies remain)
- Mean-centers ratings per user to remove rating bias
- Saves raw and normalized matrices to `output/`

### Why mean-centering?
Some users consistently rate high (e.g. always 4-5) while others rate low (e.g. always 1-3). Mean-centering removes this bias so similarity is based on relative preferences not absolute scores.

---

## Step 3: Item Similarity Matrix
Computes cosine similarity between all 939 filtered movies.

### What it does
- Loads the normalized ratings matrix from `output/`
- Transposes it so movies are rows and users are columns
- Fills remaining NaN values with 0 for similarity computation only
- Computes cosine similarity between every pair of movies
- Produces a 939 × 939 similarity matrix
- Saves matrix and movie title lookup to `output/`

### Why cosine similarity?
Cosine similarity measures the angle between two vectors regardless of magnitude. Two movies are similar if users who rated one tended to rate the other similarly — even if one movie has far more ratings than the other.

---

## Step 4: Recommendation Engine
The core recommender — given up to 5 movies, suggests up to 10 more.

### Algorithm
Item-item collaborative filtering using cosine similarity. Two movies are considered similar if users who liked one also liked the other. This captures latent patterns in real user behaviour that genre-only methods might miss.

### Success Criteria & Thresholds
- **Minimum Similarity**: 0.05 to filter out weak matches.
- **Explanations**: Every recommendation includes a reason, prioritizing genre overlap.

---

## Step 5: Interactive CLI
The main entry point — runs the full recommendation engine interactively.

```bash
python -m src.cli
```

---

## Running the Test Suite
```bash
# Run all tests
pytest tests/ -v
```

---

## Known Limitations & Assumptions
- **Dataset Age**: Based on 1997/1998 ratings (MovieLens 100K).
- **Sparsity**: ~93.7% of the matrix is sparse.
- **Cold Start**: Cannot recommend movies with < 20 ratings.
- **Exact Matching**: CLI requires exact title strings (handled by Ollama in Chatbot mode).

---

## Author
- **Dataset**: MovieLens 100K by GroupLens Research, University of Minnesota
