#!/usr/bin/env python3
"""
Script description:
This script implements selected machine learning algorithms to predict the survival of Titanic passengers.

Usage:
$ python ml_train.py
"""

import pandas as pd

# --- Model selection ---
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# --- Training and evaluation ---
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from preprocess import get_data
from transform import transform_data

def run_experiments():
    # Load and preprocess data
    train_dataset = get_data('data/train.csv', 'train')
    test_dataset = get_data('data/test.csv', 'test')

    X_train, X_valid, X_test, y_train, y_valid = transform_data(train_dataset, test_dataset)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]

    # Define models and parameter grids for GridSearchCV
    models = {
        'random_forest': RandomForestClassifier(random_state = 2025),
        'log_reg': LogisticRegression(random_state = 2025),
        'naive_bayes': GaussianNB(),
        'svm': SVC(probability = True),
        'decision_tree': DecisionTreeClassifier(random_state = 2025),
        'xgboost': XGBClassifier(random_state = 2025, use_label_encoder = False, eval_metric = 'logloss')
    }

    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'log_reg': {
            'penalty': ['l1', 'l2'],
            'max_iter': [100, 150, 200, 300],
            'C': [0.5, 1.0, 2.5, 5.0],
            'solver': ['liblinear']
        },
        'naive_bayes': {
            'var_smoothing': [1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
        },
        'svm': {
            'C': [0.5, 1.0, 2.5, 5.0],
            'kernel': ['rbf'],
            'max_iter': [100, 200, 300, 500, 600],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'decision_tree': {
            'splitter': ['best', 'random'],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.05, 0.1, 0.3, 0.5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 10]
        }
    }

    # Train models
    best_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        
        grid = GridSearchCV(
            estimator = model,
            param_grid = param_grids.get(name, {}),
            cv = 5,
            scoring = 'accuracy'
        )
        
        grid.fit(X_train, y_train)
        
        best_models[name] = {
            'best_estimator': grid.best_estimator_,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_
        }
        
        print(f"Best score for {name}: {grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}\n")

    #TODO: Log best score and hyperparameters for each model to file or dataframe

    # Evaluate best model from GridSearchCV
    best_model_name, best_model_info = max(best_models.items(), key = lambda item: item[1]['best_score'])
    best_model = best_model_info['best_estimator']

    y_preds = best_model.predict(X_valid)
    print(f"Classification report for {best_model_name}:")
    print(classification_report(y_valid, y_preds))

    # Define and evaluate voting classifier
    sorted_models = sorted(best_models.items(), key = lambda item: item[1]['best_score'], reverse = True)
    top_3 = sorted_models[:3]
    top_3_estimators = [(name, info['best_estimator']) for name, info in top_3]

    voting_clf = VotingClassifier(
        estimators = top_3_estimators,
        voting = 'soft'
    )

    voting_clf.fit(X_train, y_train)

    y_preds_voting = voting_clf.predict(X_valid)

    print(f"VotingClassifier Accuracy: {accuracy_score(y_valid, y_preds_voting):.4f}")
    print("Classification report (VotingClassifier):")
    print(classification_report(y_valid, y_preds_voting))

    # Generate submission
    """test_preds = voting_clf.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_dataset.index,
                        'Survived': test_preds})
    output.to_csv('submission_voting_clf.csv', index = False)"""

if __name__ == "__main__":
    run_experiments()
