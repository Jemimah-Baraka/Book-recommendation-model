
import pandas as pd


# Load users dataset
def load_users():

    users = pd.read_csv(
        "data/Users.csv",
        encoding="latin-1"
    )

    users.columns = users.columns.str.strip()

    # Convert age into numeric values
    users["Age"] = pd.to_numeric(
        users["Age"],
        errors="coerce"
    )

    return users


# Find a user using their ID
def find_user(users, user_id):

    user = users[
        users["User-ID"] == user_id
    ]

    if user.empty:
        return None

    return user.iloc[0]


# Display user information
def display_user(user):

    print("\nUSER INFORMATION")
    print("-" * 40)

    print("User ID:", user["User-ID"])
    print("Location:", user["Location"])
    print("Age:", user["Age"])

