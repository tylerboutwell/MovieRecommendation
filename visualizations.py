
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def create_release_year_chart(df):
    """Show the number of movies released each year."""

    data = df.copy()

    data["release_date"] = pd.to_datetime(
        data["release_date"], errors="coerce"
    )

    data["year"] = data["release_date"].dt.year

    yearly_counts = (
        data["year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        yearly_counts.index,
        yearly_counts.values,
        marker="o"
    )

    ax.set_title("Movies Released by Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Movies")

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=10, integer=True)
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def create_rating_chart(df):
    """Show the distribution of movie ratings."""

    ratings = pd.to_numeric(
        df["vote_average"], errors="coerce"
    ).dropna()

    # Only include movies with actual ratings
    ratings = ratings[ratings > 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        ratings,
        bins=20,
        edgecolor="black"
    )

    ax.set_title("Movie Rating Distribution")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Number of Movies")

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def create_votes_rating_chart(df):
    """Show the relationship between movie ratings and vote counts."""

    data = df[["vote_average", "vote_count"]].copy()

    data["vote_average"] = pd.to_numeric(
        data["vote_average"], errors="coerce"
    )

    data["vote_count"] = pd.to_numeric(
        data["vote_count"], errors="coerce"
    )

    data = data.dropna()

    # Remove movies without ratings or votes
    data = data[
        (data["vote_average"] > 0) &
        (data["vote_count"] > 0)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        data["vote_count"],
        data["vote_average"],
        alpha=0.5
    )

    ax.set_title("Movie Ratings vs. Vote Counts")
    ax.set_xlabel("Number of Votes (Log Scale)")
    ax.set_ylabel("Average Rating")

    # Vote counts can vary significantly
    ax.set_xscale("log")

    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
