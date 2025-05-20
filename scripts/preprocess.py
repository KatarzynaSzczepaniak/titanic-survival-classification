#!/usr/bin/env python3
"""
Script description:
This script performs raw data preprocessing and engineering.

Usage:
$ python preprocess.py
"""

# --- Imports ---

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (9, 6)
sns.set_theme(style="whitegrid")
sns.set_palette("deep")


def _extract_title(row: pd.Series) -> pd.Series:
    """
    Extract and normalize the title from a passenger's name,
    clean the name, and create an Alias indicator.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        pd.Series: Updated row with 'Title', cleaned 'Name' and 'Alias' columns.
    """
    try:
        surname, title_name_alias = row["Name"].split(", ")
        title, name_alias = title_name_alias.split(". ", 1)
        name = f"{surname}, {name_alias.strip()}"
        has_alias = "(" in name_alias and ")" in name_alias
    except ValueError:
        title = "Unknown"
        name = row["Name"]
        has_alias = False

    # Normalize title
    if title == "Dr":
        row["Title"] = "Mrs" if row["Sex"] == "female" else "Mr"
    elif title in ["Mr", "Dr", "Rev", "Major", "Col", "Capt", "Sir", "Don", "Jonkheer"]:
        row["Title"] = "Mr"
    elif title in ["Dona", "Mrs", "Mme", "the Countess"]:
        row["Title"] = "Mrs"
    elif title in ["Miss", "Ms", "Mlle", "Lady"]:
        row["Title"] = "Miss"
    elif title == "Master":
        row["Title"] = "Master"
    else:
        row["Title"] = title

    row["Name"] = name
    row["Alias"] = int(has_alias)

    return row


def _extract_deck(row: pd.Series) -> pd.Series:
    """
    Extract the deck letter from the cabin identifier.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        pd.Series: Updated row with 'Deck' and modified 'Cabin'.
    """

    cabin = str(row["Cabin"])
    deck_id = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]

    row["Deck"] = "Unknown"

    for deck in deck_id:
        if cabin.startswith(deck):
            row["Deck"] = deck
            row["Cabin"] = cabin.replace(deck, "").strip()
            break

    return row


def _full_port_of_embarkation(row: pd.Series) -> pd.Series:
    """
    Replace port code with full name.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        pd.Series: Updated row with full port name in 'Embarked'.
    """

    port_dict = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    row["Embarked"] = port_dict.get(row["Embarked"], "Unknown")

    return row


def get_data(path_to_dataset: str, stage: str) -> pd.DataFrame:
    """
    Load and preprocess Titanic dataset.

    Args:
        path_to_dataset (str): Path to the CSV dataset file.
        stage (str): One of ['train', 'test']. Determines which columns to return.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """

    df = pd.read_csv(path_to_dataset, index_col="PassengerId")

    # Fill missing values
    group_cols = ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]
    df["Age"] = df["Age"].fillna(df.groupby(group_cols)["Age"].transform("mean"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df[["Cabin", "Embarked"]] = df[["Cabin", "Embarked"]].fillna("Unknown")

    # Rename columns
    df = df.rename(
        columns={
            "Pclass": "Ticket Class",
            "SibSp": "# of Siblings/Spouses",
            "Parch": "# of Parents/Children",
        }
    )

    # Feature engineering
    df = df.apply(_extract_title, axis=1)
    df = df.apply(_extract_deck, axis=1)
    df = df.apply(_full_port_of_embarkation, axis=1)

    df["Family Size"] = df["# of Siblings/Spouses"] + df["# of Parents/Children"] + 1
    df["Is_child"] = (df["Age"] < 16).astype(int)
    df["Is_alone"] = (df["Family Size"] == 1).astype(int)
    df["Fare per Person"] = df["Fare"] / df["Family Size"]

    # Reorder columns
    base_cols = [
        "Title",
        "Name",
        "Alias",
        "Sex",
        "Age",
        "Is_child",
        "# of Siblings/Spouses",
        "# of Parents/Children",
        "Family Size",
        "Is_alone",
        "Ticket",
        "Ticket Class",
        "Fare",
        "Fare per Person",
        "Deck",
        "Cabin",
        "Embarked",
    ]
    if stage == "train":
        cols = ["Survived"] + base_cols
    elif stage == "test":
        cols = base_cols
    else:
        raise ValueError("Stage must be either 'train' or 'test'.")

    return df[cols]


def plot_survivability(
    df: pd.DataFrame, by: str, bins: int = None, kde: bool = False
) -> None:
    """
    Plots survivability distribution by a given feature.

    Args:
        df (pd.DataFrame): The Titanic dataset.
        by (str): Column name to plot survivability against.
        bins (int, optional): Number of bins (only for numeric data).
            Defaults to None.
        kde (bool, optional): Whether to include KDE curve (only for numeric data).
            Defaults to False.
    """

    sns.histplot(
        data=df, x=by, hue="Survived", bins=bins, kde=kde, multiple="dodge", shrink=0.8
    )
    plt.title(f"Survivability by {by}")
    plt.xticks(rotation=45 if df[by].dtype == "object" else 0)
    plt.tight_layout()
    plt.show()


def main():
    train_dataset = get_data("data/train.csv", "train")

    # Basic overview
    print(train_dataset.shape)
    print(train_dataset.info())
    print(train_dataset.describe(include="all"))
    print(train_dataset.isnull().sum(), end="\n\n")

    # Target distribution
    sns.countplot(x="Survived", data=train_dataset)
    plt.title("Survival Counts")
    plt.show()

    print(train_dataset["Survived"].value_counts(normalize=True))

    # Correlations
    numeric_cols = train_dataset.select_dtypes(include="number")
    sns.heatmap(numeric_cols.corr(), fmt=".3f", annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Survival by Engineered Features
    train_dataset.groupby("Is_child")["Survived"].mean().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Survival Rate by Child Status")
    plt.show()

    train_dataset.groupby("Is_alone")["Survived"].mean().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Survival Rate by Being Alone")
    plt.show()

    # Survivability by Select Features
    plot_survivability(train_dataset, "Sex")
    plot_survivability(train_dataset, "Age", bins=20, kde=True)
    plot_survivability(train_dataset, "Title")
    plot_survivability(train_dataset, "Fare per Person", bins=20, kde=True)


if __name__ == "__main__":
    main()
