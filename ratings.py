
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# Load ratings dataset
def load_ratings():

    ratings = pd.read_csv(
        "data/Ratings.csv",
        encoding="latin-1",
        dtype={"ISBN": str}
    )

    ratings.columns = ratings.columns.str.strip()

    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["User-ID", "ISBN", "Book-Rating"]
    )

    # Keep explicit ratings only (1 to 10)
    ratings = ratings[
        ratings["Book-Rating"].between(1, 10)
    ].copy()

    ratings = ratings.drop_duplicates(
        subset=["User-ID", "ISBN"]
    )

    return ratings


# Build a sparse user-book rating matrix
def build_user_item_matrix(ratings):

    user_ids = ratings["User-ID"].unique()
    book_ids = ratings["ISBN"].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    book_to_index = {
        book_id: index
        for index, book_id in enumerate(book_ids)
    }

    row = ratings["User-ID"].map(user_to_index)
    column = ratings["ISBN"].map(book_to_index)

    rating_values = ratings["Book-Rating"].values

    matrix = csr_matrix(
        (
            rating_values,
            (row, column)
        ),
        shape=(len(user_ids), len(book_ids))
    )

    return matrix, user_to_index, book_to_index


# Recommend books based on similar users
def recommend_for_user(
    user_id,
    ratings,
    books,
    user_item_matrix,
    user_to_index,
    book_to_index,
    number=5
):

    # Check whether user exists in the rating matrix
    if user_id not in user_to_index:
        return None

    user_index = user_to_index[user_id]

    # Calculate similarity between this user and other users
    similarities = cosine_similarity(
        user_item_matrix[user_index],
        user_item_matrix
    ).ravel()

    # Get the most similar users
    similar_indices = similarities.argsort()[::-1]

    similar_users = []

    for index in similar_indices:

        if index == user_index:
            continue

        similarity = similarities[index]

        if similarity > 0:
            similar_users.append((index, similarity))

        if len(similar_users) == 20:
            break

    if not similar_users:
        return None

    # Convert similar user indexes back into user IDs
    index_to_user = {
        index: user
        for user, index in user_to_index.items()
    }

    similar_user_scores = {
        index_to_user[index]: similarity
        for index, similarity in similar_users
    }

    # Get ratings from similar users
    similar_ratings = ratings[
        ratings["User-ID"].isin(similar_user_scores.keys())
    ].copy()

    similar_ratings["Similarity"] = (
        similar_ratings["User-ID"].map(similar_user_scores)
    )

    # Exclude books already rated by the current user
    already_rated = set(
        ratings.loc[
            ratings["User-ID"] == user_id,
            "ISBN"
        ]
    )

    similar_ratings = similar_ratings[
        ~similar_ratings["ISBN"].isin(already_rated)
    ]

    if similar_ratings.empty:
        return None

    # Calculate weighted average ratings
    similar_ratings["Weighted-Rating"] = (
        similar_ratings["Book-Rating"] *
        similar_ratings["Similarity"]
    )

    recommendations = (
        similar_ratings.groupby("ISBN")
        .agg(
            weighted_sum=("Weighted-Rating", "sum"),
            total_weight=("Similarity", "sum"),
            number_of_ratings=("Book-Rating", "count")
        )
        .reset_index()
    )

    recommendations["Predicted-Rating"] = (
        recommendations["weighted_sum"] /
        recommendations["total_weight"]
    )

    # Sort by predicted rating
    recommendations = recommendations.sort_values(
        "Predicted-Rating",
        ascending=False
    )

    # Attach book details
    recommendations = recommendations.merge(
        books[
            ["ISBN", "Book-Title", "Book-Author"]
        ],
        on="ISBN",
        how="inner"
    )

    recommendations = recommendations.head(number)

    return recommendations[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Predicted-Rating",
            "number_of_ratings"
        ]
    ]


# Recommend popular books for users without enough rating history
def popular_books(ratings, books, number=5):

    popular = (
        ratings.groupby("ISBN")
        .agg(
            average_rating=("Book-Rating", "mean"),
            number_of_ratings=("Book-Rating", "count")
        )
        .reset_index()
    )

    # Require at least 5 ratings to reduce the influence of
    # books with very few ratings
    popular = popular[
        popular["number_of_ratings"] >= 5
    ]

    popular = popular.sort_values(
        ["average_rating", "number_of_ratings"],
        ascending=False
    )

    popular = popular.merge(
        books[
            ["ISBN", "Book-Title", "Book-Author"]
        ],
        on="ISBN",
        how="inner"
    )

    return popular[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "average_rating",
            "number_of_ratings"
        ]
    ].head(number)

