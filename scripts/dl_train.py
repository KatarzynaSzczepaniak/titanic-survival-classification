#!/usr/bin/env python3
"""
Script description:
This script runs deep learning experiments for Titanic survival prediction
using Keras or PyTorch, with optional ensembling of the top models.

Usage:
$ python dl_train.py --framework keras
$ python dl_train.py --framework pytorch --ensemble
"""

# --- Imports ---

# Standard libraries
import argparse
from typing import Dict, Any

# Third-party libraries
import numpy as np
import pandas as pd

# Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import BCELoss

# Scikit-learn
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from preprocess import get_data
from transform import transform_data


def run_keras_gridsearch() -> Dict[str, Dict[str, Any]]:
    """
    Run grid search on models defined with Keras.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary holding instance of the model
            and score for each trained model.
    """
    pass


def run_pytorch_gridsearch() -> Dict[str, Dict[str, Any]]:
    """
    Run grid search on models defined with PyTorch.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary holding instance of the model
            and score for each trained model.
    """
    pass


def run_ensemble(best_models: Dict[str, Dict[str, Any]]) -> VotingClassifier:
    """
    Select the top 3 models from best_models dict and ensemble them via soft voting.

    Args:
        best_models (Dict[str, Dict[str, Any]]): A dict where keys are model names
            and values are dicts containing 'model' and 'score'.

    Returns:
        VotingClassifier: A trained VotingClassifier instance.
    """
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train DL models on Titanic data.")
    parser.add_argument(
        "--framework",
        choices=["keras", "pytorch"],
        required=True,
        help="Choose backend framework.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="If set, ensemble top 3 models from grid search.",
    )
    return parser.parse_args()


def run_experiment(framework: str, do_ensemble: bool):
    # TODO: Implement deep learning training pipeline
    if framework == "keras":
        best_models = run_keras_gridsearch()  # returns dict of best models
    else:
        best_models = run_pytorch_gridsearch()

    if do_ensemble:
        run_ensemble(best_models)  # pick top 3, average predictions, evaluate


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.framework, args.ensemble)
