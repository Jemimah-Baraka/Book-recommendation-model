
# ==========================================
# BOOK RECOMMENDATION SYSTEM
# ==========================================

from books import (
    load_books,
    build_content_model,
    recommend_similar_books
)

from users import (
    load_users,
    find_user,
    display_user
)

from ratings import (
    load_ratings,
    build_user_item_matrix,
    recommend_for_user,
    popular_books
)


# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("\nLoading datasets...")

books = load_books()
users = load_users()
ratings = load_ratings()

print("\nDatasets loaded successfully!")

print("Total books:", len(books))
print("Total users:", len(users))
print("Total ratings:", len(ratings))


# ==========================================
# 2. BUILD RECOMMENDATION MODELS
# ==========================================

print("\nBuilding recommendation models...")

# Content-based recommendation model
tfidf_matrix, content_model = build_content_model(books)

print("Content-based model ready!")

# Collaborative filtering model
user_item_matrix, user_to_index, book_to_index = (
    build_user_item_matrix(ratings)
)

print("Collaborative filtering model ready!")

print("\nAll recommendation models are ready!")


# ==========================================
# 3. DISPLAY RECOMMENDATIONS
# ==========================================

def display_recommendations(recommendations):

    if recommendations is None or recommendations.empty:

        print("\nNo recommendations found.")

        return

    print("\nRECOMMENDED BOOKS")
    print("=" * 60)

    for position, (_, book) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(f"\n{position}. {book['Book-Title']}")

        print("Author:", book["Book-Author"])

        # Display content-based similarity
        if "Similarity" in recommendations.columns:

            print(
                "Similarity:",
                str(book["Similarity"]) + "%"
            )

        # Display predicted rating
        if "Predicted-Rating" in recommendations.columns:

            print(
                "Predicted rating:",
                round(book["Predicted-Rating"], 2),
                "/ 10"
            )

        # Display average rating
        if "average_rating" in recommendations.columns:

            print(
                "Average rating:",
                round(book["average_rating"], 2),
                "/ 10"
            )

        # Display number of ratings
        if "number_of_ratings" in recommendations.columns:

            print(
                "Number of ratings:",
                book["number_of_ratings"]
            )


# ==========================================
# 4. MAIN MENU
# ==========================================

def main():

    while True:

        print("\n")
        print("=" * 55)
        print("          BOOK RECOMMENDATION SYSTEM")
        print("=" * 55)

        print("\n1. Recommend books based on a book title")
        print("2. Get personalised recommendations using User ID")
        print("3. View popular books")
        print("4. View user information")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()


        # ==================================
        # OPTION 1: CONTENT-BASED FILTERING
        # ==================================

        if choice == "1":

            title = input(
                "\nEnter the title of a book you like: "
            )

            recommendations = recommend_similar_books(
                books=books,
                tfidf_matrix=tfidf_matrix,
                model=content_model,
                title=title,
                number=5
            )

            if recommendations is None or recommendations.empty:

                print("\nBook not found in the dataset.")

                print(
                    "Please check the spelling "
                    "or try another book title."
                )

            else:

                display_recommendations(recommendations)


        # ==================================
        # OPTION 2: COLLABORATIVE FILTERING
        # ==================================

        elif choice == "2":

            try:

                user_id = int(
                    input("\nEnter your User ID: ")
                )

                # Find the user
                user = find_user(users, user_id)

                if user is None:

                    print("\nUser ID not found in the dataset.")

                else:

                    # Display user information
                    display_user(user)

                    # Generate personalised recommendations
                    recommendations = recommend_for_user(
                        user_id,
                        ratings,
                        books,
                        user_item_matrix,
                        user_to_index,
                        book_to_index,
                        number=5
                    )

                    # If personalised recommendations are unavailable
                    if recommendations is None or recommendations.empty:

                        print(
                            "\nNot enough information "
                            "for personalised recommendations."
                        )

                        print("\nShowing popular books instead.")

                        recommendations = popular_books(
                            ratings,
                            books,
                            number=5
                        )

                    display_recommendations(recommendations)

            except ValueError:

                print("\nPlease enter a valid numerical User ID.")


        # ==================================
        # OPTION 3: POPULAR BOOKS
        # ==================================

        elif choice == "3":

            print("\nFinding popular books...")

            recommendations = popular_books(
                ratings,
                books,
                number=10
            )

            display_recommendations(recommendations)


        # ==================================
        # OPTION 4: USER INFORMATION
        # ==================================

        elif choice == "4":

            try:

                user_id = int(
                    input("\nEnter your User ID: ")
                )

                user = find_user(users, user_id)

                if user is None:

                    print("\nUser not found.")

                else:

                    display_user(user)

            except ValueError:

                print("\nPlease enter a valid numerical User ID.")


        # ==================================
        # OPTION 5: EXIT
        # ==================================

        elif choice == "5":

            print("\nThank you for using the Book Recommendation System!")

            break


        # ==================================
        # INVALID OPTION
        # ==================================

        else:

            print("\nInvalid choice.")

            print("Please select a number between 1 and 5.")


# ==========================================
# 5. RUN THE PROGRAM
# ==========================================

if __name__ == "__main__":

    main()
