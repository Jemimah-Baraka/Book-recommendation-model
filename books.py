
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ==========================================
# 1. LOAD AND PREPARE BOOKS
# ==========================================

def load_books():

    books = pd.read_csv(
        "data/Books.csv",
        encoding="latin-1",
        dtype={
            "ISBN": str,
            "Year-Of-Publication": str
        }
    )

    # Remove unnecessary spaces from column names
    books.columns = books.columns.str.strip()

    # Select the columns needed
    books = books[
        ["ISBN", "Book-Title", "Book-Author", "Publisher"]
    ].copy()

    # Handle missing values
    for column in ["Book-Title", "Book-Author", "Publisher"]:
        books[column] = (
            books[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Remove books without titles
    books = books[books["Book-Title"] != ""]

    # Remove duplicate ISBNs
    books = books.drop_duplicates(subset="ISBN")

    # Combine book information
    books["combined_features"] = (
        books["Book-Title"] + " " +
        books["Book-Author"] + " " +
        books["Publisher"]
    )

    books = books.reset_index(drop=True)

    return books


# ==========================================
# 2. BUILD CONTENT-BASED MODEL
# ==========================================

def build_content_model(books):

    # Convert text into numerical features
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = tfidf.fit_transform(
        books["combined_features"]
    )

    # Use Nearest Neighbors instead of creating
    # a full similarity matrix
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(tfidf_matrix)

    return tfidf_matrix, model


# ==========================================
# 3. FIND A BOOK
# ==========================================

def find_book(books, title):

    search_title = title.strip().casefold()

    matches = books[
        books["Book-Title"].str.casefold() == search_title
    ]

    if matches.empty:
        return None

    return matches.index[0]


# ==========================================
# 4. RECOMMEND SIMILAR BOOKS
# ==========================================

def recommend_similar_books(
    books,
    tfidf_matrix,
    model,
    title,
    number=5
):


    book_index = find_book(books, title)

    if book_index is None:
        return None

    # Find the nearest books
    distances, indices = model.kneighbors(
        tfidf_matrix[book_index],
        n_neighbors=min(number + 1, len(books))
    )

    recommendations = []

    for distance, index in zip(distances[0], indices[0]):

        # Exclude the selected book
        if index == book_index:
            continue

        book = books.iloc[index]

        # Convert cosine distance into similarity percentage
        similarity = (1 - distance) * 100

        recommendations.append({
            "ISBN": book["ISBN"],
            "Book-Title": book["Book-Title"],
            "Book-Author": book["Book-Author"],
            "Similarity": round(similarity, 2)
        })

        if len(recommendations) == number:
            break

    return pd.DataFrame(recommendations)


