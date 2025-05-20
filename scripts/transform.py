#!/usr/bin/env python3
"""
Script description:
This script prepares data for ML/DL pipeline by scaling and encoding the features,
and splitting the data into train/val/test sets.

Usage:
$ python3 transform.py
"""

# --- Imports ---

# Standard libraries
from typing import List, Tuple, Optional

# Third-party libraries
import numpy as np
import pandas as pd

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def transform_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "Survived",
    drop_cols: Optional[List[str]] = None,
    random_state: int = 2025,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms train and test DataFrames into ML-ready NumPy arrays.

    Args:
        train_df (pd.DataFrame): Preprocessed training DataFrame.
        test_df (pd.DataFrame): Preprocessed test DataFrame.
        target (str): Name of the target column.
        drop_cols (List[str], optional): List of columns to drop from features
            (e.g., 'Name', 'Ticket').
        random_state (int, optional): Seed for train/validation split.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
            containing:
            - X_train: Transformed training features.
            - X_valid: Transformed validation features.
            - X_test: Transformed test features.
            - y_train: Target labels for training.
            - y_valid: Target labels for validation.
    """
    if drop_cols is None:
        drop_cols = ["Name", "Ticket", "Cabin"]

    # Prepare target and features
    X = train_df.drop(columns=drop_cols + [target])
    y = train_df[target]
    X_test = test_df.drop(columns=drop_cols)

    # Define column transformer
    preprocessor = make_column_transformer(
        (
            make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler()),
            make_column_selector(dtype_include=np.number),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
            make_column_selector(dtype_include=["object", "category"]),
        ),
    )

    # Split training data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, stratify=y, random_state=random_state
    )

    # Fit preprocessor on training data, transform all
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)

    return X_train, X_valid, X_test, y_train.to_numpy(), y_valid.to_numpy()


def main():
    from preprocess import get_data

    train_dataset = get_data("data/train.csv", "train")
    test_dataset = get_data("data/test.csv", "test")

    X_train, X_valid, X_test, y_train, y_valid = transform_data(
        train_dataset, test_dataset
    )

    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]

    print(
        f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}"
    )


if __name__ == "__main__":
    main()
