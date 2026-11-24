import pandas as pd

DATA_PATH = "data/Treanding_movies.csv"


def load_movies():
    """Load and clean the movie dataset (used by both app and recommender)."""
    movies = pd.read_csv(DATA_PATH)

    # Drop the unnamed index column
    movies = movies.drop(columns=["Unnamed: 0"], errors="ignore")

    # Parse dates; keep only released movies
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies = movies[movies["release_date"] <= pd.Timestamp.today()]

    # Keep movies with a usable overview and at least one vote
    movies = movies.dropna(subset=["title", "overview"])
    movies = movies[movies["overview"].str.strip().str.len() >= 30]
    movies = movies[movies["vote_count"] > 0]

    # Remove duplicates
    movies = movies.drop_duplicates(subset=["id"])
    movies = movies.drop_duplicates(subset=["title"], keep="first")

    # Back to plain date strings so the app displays them cleanly
    movies["release_date"] = movies["release_date"].dt.strftime("%Y-%m-%d")

    return movies.reset_index(drop=True)