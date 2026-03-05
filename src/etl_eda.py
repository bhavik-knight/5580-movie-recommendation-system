"""
ETL and Exploratory Data Analysis (EDA) module.
Handles loading of the MovieLens 100K dataset, performs sanity checks,
and generates summary statistics and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    U_DATA_PATH, U_ITEM_PATH, U_USER_PATH, U_GENRE_PATH, 
    U_OCCUPATION_PATH, EDA_SUMMARY_FILE, GENRE_COLUMNS,
    U_DATA_NAMES, U_ITEM_NAMES, U_USER_NAMES, U_GENRE_NAMES,
    U_OCCUPATION_NAMES, RATING_DIST_PLOT, GENRE_DIST_PLOT,
    USER_AGE_PLOT, USER_GENDER_PLOT, OUTPUT_DIR
)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all raw data files from the MovieLens 100K dataset using configuration paths.
    
    Returns:
        tuple: A tuple containing (u_data, u_item, u_user, u_genre, u_occupation) as pandas DataFrames.
    """
    # Load u.data (ratings)
    u_data = pd.read_csv(U_DATA_PATH, sep="\t", names=U_DATA_NAMES)

    # Load u.item (movie metadata)
    u_item = pd.read_csv(U_ITEM_PATH, sep="|", names=U_ITEM_NAMES, encoding="ISO-8859-1")

    # Load u.user (user demographics)
    u_user = pd.read_csv(U_USER_PATH, sep="|", names=U_USER_NAMES)

    # Load u.genre (genre lookup)
    u_genre = pd.read_csv(U_GENRE_PATH, sep="|", names=U_GENRE_NAMES)

    # Load u.occupation (occupation lookup)
    u_occupation = pd.read_csv(U_OCCUPATION_PATH, names=U_OCCUPATION_NAMES)

    return u_data, u_item, u_user, u_genre, u_occupation

def perform_sanity_checks(u_data: pd.DataFrame, u_item: pd.DataFrame, u_user: pd.DataFrame, 
                          u_genre: pd.DataFrame, u_occupation: pd.DataFrame) -> str:
    """
    Perform structural and consistency sanity checks on the loaded dataframes.
    
    Args:
        u_data (pd.DataFrame): Ratings dataframe.
        u_item (pd.DataFrame): Movies dataframe.
        u_user (pd.DataFrame): Users dataframe.
        u_genre (pd.DataFrame): Genres dataframe.
        u_occupation (pd.DataFrame): Occupations dataframe.

    Returns:
        str: A formatted string summarizing the results of the sanity checks.
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

    # Consistency check: Ensure all movie_ids in ratings exist in the movie lookup
    data_movie_ids = set(u_data['movie_id'].unique())
    item_movie_ids = set(u_item['movie_id'].unique())
    is_consistent = data_movie_ids.issubset(item_movie_ids)
    results.append(f"\nMovie ID Consistency (u.data subset of u.item): {is_consistent}")
    
    if not is_consistent:
        missing = data_movie_ids - item_movie_ids
        results.append(f"  Missing IDs in u.item (first 10): {list(missing)[:10]}...")

    return "\n".join(results)

def run_eda_data(u_data: pd.DataFrame, u_item: pd.DataFrame) -> str:
    """
    Perform Exploratory Data Analysis on the ratings dataset.
    Generates rating distribution statistics and visualizations.
    
    Args:
        u_data (pd.DataFrame): Ratings dataframe.
        u_item (pd.DataFrame): Movies dataframe for title merging.

    Returns:
        str: A summary of ratings-related EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.data (Ratings) ===")
    
    # Rating distribution statistics
    rating_counts = u_data['rating'].value_counts().sort_index()
    results.append(f"Rating distribution:\n{rating_counts.to_string()}")
    
    # Save rating distribution plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=u_data, hue='rating', palette='viridis', legend=False)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(RATING_DIST_PLOT)
    plt.close()
    
    # User-level rating stats
    user_stats = u_data.groupby('user_id')['rating'].count()
    results.append(f"\nRatings per user:")
    results.append(f"  Min: {user_stats.min()}")
    results.append(f"  Max: {user_stats.max()}")
    results.append(f"  Mean: {user_stats.mean():.2f}")
    
    # Movie-level rating stats
    movie_stats = u_data.groupby('movie_id')['rating'].agg(['count', 'mean'])
    results.append(f"\nRatings per movie:")
    results.append(f"  Min ratings: {movie_stats['count'].min()}")
    results.append(f"  Max ratings: {movie_stats['count'].max()}")
    results.append(f"  Mean ratings: {movie_stats['count'].mean():.2f}")
    results.append(f"  Mean average rating: {movie_stats['mean'].mean():.2f}")

    # Identify top 10 most rated movies
    top_10_rated = movie_stats.sort_values(by='count', ascending=False).head(10)
    top_10_rated_with_titles = top_10_rated.merge(u_item[['movie_id', 'title']], on='movie_id')
    results.append(f"\nTop 10 most rated movies:\n{top_10_rated_with_titles[['title', 'count']].to_string(index=False)}")

    # Identify top 10 highest average rated movies (min 50 ratings for significance)
    top_10_avg = movie_stats[movie_stats['count'] >= 50].sort_values(by='mean', ascending=False).head(10)
    top_10_avg_with_titles = top_10_avg.merge(u_item[['movie_id', 'title']], on='movie_id')
    results.append(f"\nTop 10 highest average rated movies (min 50 ratings):\n{top_10_avg_with_titles[['title', 'mean']].to_string(index=False)}")

    return "\n".join(results)

def run_eda_item(u_item: pd.DataFrame) -> str:
    """
    Perform Exploratory Data Analysis on the movie metadata dataset.
    Generates genre distribution statistics and visualizations.
    
    Args:
        u_item (pd.DataFrame): Movies dataframe.

    Returns:
        str: A summary of movie-related EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.item (Movies) ===")
    
    # Genre frequency across the dataset
    genre_counts = u_item[GENRE_COLUMNS].sum().sort_values(ascending=False)
    results.append(f"Genre distribution:\n{genre_counts.to_string()}")
    
    # Save genre distribution plot
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(GENRE_DIST_PLOT)
    plt.close()
    
    # Multi-genre analysis
    movies_per_genre_count = u_item[GENRE_COLUMNS].sum(axis=1)
    multi_genre_count = (movies_per_genre_count > 1).sum()
    results.append(f"\nMovies belonging to multiple genres: {multi_genre_count}")
    
    # Integrity check for unknown genres
    unknown_count = u_item['unknown'].sum()
    results.append(f"Movies with unknown=1: {unknown_count}")

    return "\n".join(results)

def run_eda_user(u_user: pd.DataFrame) -> str:
    """
    Perform Exploratory Data Analysis on the user demographics dataset.
    Generates age, gender, and occupation statistics and visualizations.
    
    Args:
        u_user (pd.DataFrame): Users dataframe.

    Returns:
        str: A summary of user-related EDA findings.
    """
    results = []
    results.append("\n=== EDA on u.user (Users) ===")
    
    # Age distribution metrics
    results.append(f"Age distribution:")
    results.append(f"  Min: {u_user['age'].min()}")
    results.append(f"  Max: {u_user['age'].max()}")
    results.append(f"  Mean: {u_user['age'].mean():.2f}")
    
    # Save age distribution histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(u_user['age'], bins=20, kde=True, color='olive')
    plt.title('User Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(USER_AGE_PLOT)
    plt.close()
    
    # Gender breakdown analysis
    gender_counts = u_user['gender'].value_counts()
    results.append(f"\nGender breakdown:\n{gender_counts.to_string()}")
    
    # Save gender distribution pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
    plt.title('User Gender Distribution')
    plt.savefig(USER_GENDER_PLOT)
    plt.close()
    
    # Top occupations represented in the user base
    occ_counts = u_user['occupation'].value_counts().head(10)
    results.append(f"\nTop 10 occupations:\n{occ_counts.to_string()}")

    return "\n".join(results)

def main() -> None:
    """
    Main execution flow for the ETL and EDA process.
    Orchestrates data loading, sanity checks, analysis, and persistence of results.
    """
    # Ensure the output directory project-structure exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data from configured paths
    print("--- Movie Recommendation System: ETL & EDA ---")
    print("Loading raw data files...")
    u_data, u_item, u_user, u_genre, u_occupation = load_data()
    
    # Step 2: Validate data integrity
    print("Performing sanity checks and consistency validation...")
    sanity_results = perform_sanity_checks(u_data, u_item, u_user, u_genre, u_occupation)
    
    # Step 3: Run analysis modules
    print("Analyzing ratings (u.data)...")
    data_eda_results = run_eda_data(u_data, u_item)
    
    print("Analyzing movie items (u.item)...")
    item_eda_results = run_eda_item(u_item)
    
    print("Analyzing user demographics (u.user)...")
    user_eda_results = run_eda_user(u_user)
    
    # Step 4: Aggregate findings
    full_summary = "\n".join([
        sanity_results,
        data_eda_results,
        item_eda_results,
        user_eda_results
    ])
    
    # Display summary to console
    print("\n" + full_summary)
    
    # Step 5: Save aggregated summary to disk
    print(f"\nSaving final summary to: {EDA_SUMMARY_FILE}")
    with open(EDA_SUMMARY_FILE, "w") as f:
        f.write(full_summary)
    
    print("Process complete! Visualizations have been saved to the output directory.")

if __name__ == "__main__":
    main()
