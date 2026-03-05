import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    U_DATA_PATH, U_ITEM_PATH, U_USER_PATH, U_GENRE_PATH, 
    U_OCCUPATION_PATH, EDA_SUMMARY_FILE, GENRE_COLUMNS,
    U_DATA_NAMES, U_ITEM_NAMES, U_USER_NAMES, U_GENRE_NAMES,
    U_OCCUPATION_NAMES, RATING_DIST_PLOT, GENRE_DIST_PLOT,
    USER_AGE_PLOT, USER_GENDER_PLOT, OUTPUT_DIR
)

def load_data():
    """
    Load all data files from the MovieLens 100K dataset using config.
    
    Returns:
        tuple: (data, item, user, genre, occupation) DataFrames
    """
    # u.data
    u_data = pd.read_csv(U_DATA_PATH, sep="\t", names=U_DATA_NAMES)

    # u.item
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")

    # u.user
    u_user = pd.read_csv(U_USER_PATH, sep="|", names=U_USER_NAMES)

    # u.genre
    u_genre = pd.read_csv(U_GENRE_PATH, sep="|", names=U_GENRE_NAMES)

    # u.occupation
    u_occupation = pd.read_csv(U_OCCUPATION_PATH, names=U_OCCUPATION_NAMES)

    return u_data, u_item, u_user, u_genre, u_occupation

def perform_sanity_checks(u_data, u_item, u_user, u_genre, u_occupation):
    """
    Perform basic sanity checks on the loaded data.
    
    Returns:
        str: A summary of the sanity checks.
    """
    results = []
    results.append("=== SANITY CHECKS ===")
    
    dfs = {
        "u.data": u_data,
        "u.item": u_item,
        "u.user": u_user,
        "u.genre": u_genre,
        "u.occupation": u_occupation
    }
    
    for name, df in dfs.items():
        results.append(f"\nDataFrame: {name}")
        results.append(f"  Shape: {df.shape}")
        results.append(f"  Null values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().any() else '    None'}")
        results.append(f"  Data types:\n{df.dtypes.to_string()}")

    # Consistency check
    data_movie_ids = set(u_data['movie_id'].unique())
    item_movie_ids = set(u_item['movie_id'].unique())
    is_consistent = data_movie_ids.issubset(item_movie_ids)
    results.append(f"\nMovie ID Consistency (u.data subset of u.item): {is_consistent}")
    
    if not is_consistent:
        missing = data_movie_ids - item_movie_ids
        results.append(f"  Missing IDs in u.item: {list(missing)[:10]}...")

    return "\n".join(results)

def run_eda_data(u_data, u_item):
    """
    Perform EDA on the ratings data (u.data).
    
    Returns:
        str: A summary of the EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.data (Ratings) ===")
    
    # Rating distribution
    rating_counts = u_data['rating'].value_counts().sort_index()
    results.append(f"Rating distribution:\n{rating_counts.to_string()}")
    
    # Plot rating distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=u_data, hue='rating', palette='viridis', legend=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(RATING_DIST_PLOT)
    plt.close()
    
    # Ratings per user
    user_stats = u_data.groupby('user_id')['rating'].count()
    results.append(f"\nRatings per user:")
    results.append(f"  Min: {user_stats.min()}")
    results.append(f"  Max: {user_stats.max()}")
    results.append(f"  Mean: {user_stats.mean():.2f}")
    
    # Ratings per movie
    movie_stats = u_data.groupby('movie_id')['rating'].agg(['count', 'mean'])
    results.append(f"\nRatings per movie:")
    results.append(f"  Min ratings: {movie_stats['count'].min()}")
    results.append(f"  Max ratings: {movie_stats['count'].max()}")
    results.append(f"  Mean ratings: {movie_stats['count'].mean():.2f}")
    results.append(f"  Mean average rating: {movie_stats['mean'].mean():.2f}")

    # Top 10 most rated movies
    top_10_rated = movie_stats.sort_values(by='count', ascending=False).head(10)
    top_10_rated_with_titles = top_10_rated.merge(u_item[['movie_id', 'title']], on='movie_id')
    results.append(f"\nTop 10 most rated movies:\n{top_10_rated_with_titles[['title', 'count']].to_string(index=False)}")

    # Top 10 highest average rated movies with at least 50 ratings
    top_10_avg = movie_stats[movie_stats['count'] >= 50].sort_values(by='mean', ascending=False).head(10)
    top_10_avg_with_titles = top_10_avg.merge(u_item[['movie_id', 'title']], on='movie_id')
    results.append(f"\nTop 10 highest average rated movies (min 50 ratings):\n{top_10_avg_with_titles[['title', 'mean']].to_string(index=False)}")

    return "\n".join(results)

def run_eda_item(u_item):
    """
    Perform EDA on the movie items data (u.item).
    
    Returns:
        str: A summary of the EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.item (Movies) ===")
    
    # Genre distribution
    genre_counts = u_item[GENRE_COLUMNS].sum().sort_values(ascending=False)
    results.append(f"Genre distribution:\n{genre_counts.to_string()}")
    
    # Plot genre distribution
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(GENRE_DIST_PLOT)
    plt.close()
    
    # Count of movies belonging to multiple genres
    movies_per_genre_count = u_item[GENRE_COLUMNS].sum(axis=1)
    multi_genre_count = (movies_per_genre_count > 1).sum()
    results.append(f"\nMovies belonging to multiple genres: {multi_genre_count}")
    
    # Count of movies with unknown=1
    unknown_count = u_item['unknown'].sum()
    results.append(f"Movies with unknown=1: {unknown_count}")

    return "\n".join(results)

def run_eda_user(u_user):
    """
    Perform EDA on the user demographic data (u.user).
    
    Returns:
        str: A summary of the EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.user (Users) ===")
    
    # Age distribution
    results.append(f"Age distribution:")
    results.append(f"  Min: {u_user['age'].min()}")
    results.append(f"  Max: {u_user['age'].max()}")
    results.append(f"  Mean: {u_user['age'].mean():.2f}")
    
    # Plot age distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(u_user['age'], bins=20, kde=True, color='olive')
    plt.title('User Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(USER_AGE_PLOT)
    plt.close()
    
    # Gender breakdown
    gender_counts = u_user['gender'].value_counts()
    results.append(f"\nGender breakdown:\n{gender_counts.to_string()}")
    
    # Plot gender breakdown
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
    plt.title('User Gender Distribution')
    plt.savefig(USER_GENDER_PLOT)
    plt.close()
    
    # Top 10 occupations by count
    occ_counts = u_user['occupation'].value_counts().head(10)
    results.append(f"\nTop 10 occupations:\n{occ_counts.to_string()}")

    return "\n".join(results)

def main():
    """
    Main execution flow for ETL and EDA.
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    u_data, u_item, u_user, u_genre, u_occupation = load_data()
    
    # Run analysis
    print("Performing sanity checks...")
    sanity_results = perform_sanity_checks(u_data, u_item, u_user, u_genre, u_occupation)
    
    print("Running EDA on ratings...")
    data_eda_results = run_eda_data(u_data, u_item)
    
    print("Running EDA on items...")
    item_eda_results = run_eda_item(u_item)
    
    print("Running EDA on users...")
    user_eda_results = run_eda_user(u_user)
    
    # Combine and save results
    full_summary = "\n".join([
        sanity_results,
        data_eda_results,
        item_eda_results,
        user_eda_results
    ])
    
    # Print to console
    print("\n" + full_summary)
    
    # Save to file
    print(f"\nSaving summary to {EDA_SUMMARY_FILE}...")
    with open(EDA_SUMMARY_FILE, "w") as f:
        f.write(full_summary)
    
    print("Done! Visualizations saved to output/ directory.")

if __name__ == "__main__":
    main()
