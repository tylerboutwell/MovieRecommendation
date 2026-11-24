import streamlit as st
import pandas as pd

from data_loader import load_movies
from recommender import get_recommendations
from visualizations import (
    create_release_year_chart,
    create_rating_chart,
    create_votes_rating_chart
)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

st.title("Movie Recommendation System")

st.write("Select a movie to find similar movies.")

# Load movie dataset
@st.cache_data
def get_movies():
    return load_movies()

movies = get_movies()

# Search for a movie
selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].dropna().unique()
)

# Display selected movie
if selected_movie:
    movie = movies[movies["title"] == selected_movie].iloc[0]

    st.subheader(movie["title"])
    st.write(movie["overview"])
    st.write(f"Release Date: {movie['release_date']}")
    st.write(f"Rating: {movie['vote_average']}/10")

    st.divider()

    # Get recommendations
    st.subheader("Recommended Movies")

    recommendations = get_recommendations(selected_movie)

    # Display recommendations
    for _, recommended_movie in recommendations.iterrows():
        st.write(f"### {recommended_movie['title']}")
        st.write(recommended_movie["overview"])
        st.write(f"Release Date: {recommended_movie['release_date']}")
        st.write(f"Rating: {recommended_movie['vote_average']}/10")
        st.divider()

    st.header("Movie Dataset Analysis")

    st.pyplot(create_release_year_chart(movies), width="stretch")

    st.pyplot(create_rating_chart(movies), width="stretch")

    st.pyplot(create_votes_rating_chart(movies), width="stretch")