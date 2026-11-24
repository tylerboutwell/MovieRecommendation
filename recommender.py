
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load movie dataset
movies = pd.read_csv("data/Treanding_movies.csv")

# Remove duplicate movie titles
movies = movies.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
features = vectorizer.fit_transform(
    movies["overview"].fillna("")
)

# Fit nearest-neighbor search model
neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
neighbors.fit(features)


def get_recommendations(movie_title, n=5):
    # Find selected movie
    movie_index = movies[
        movies["title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_index) == 0:
        return pd.DataFrame()

    movie_index = movies.index.get_loc(movie_index[0])

    # Find nearest neighbors
    distances, indices = neighbors.kneighbors(
        features[movie_index],
        n_neighbors=min(n + 1, len(movies))
    )

    # Exclude selected movie
    recommendations = movies.iloc[indices[0][1:]]

    # Drop duplicates by titles
    recommendations = recommendations.drop_duplicates(subset=["title"])

    return recommendations[["title", "overview", "release_date", "vote_average", "popularity"]]
