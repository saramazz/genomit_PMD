"""
Script to perform classification of nDNA vs mtDNA.
Evaluates feature selection methods, balancing techniques, and various classifiers.
"""

# Standard Library Imports
import os
import sys
import time
import json
from collections import Counter
from itertools import combinations
from datetime import datetime
import os

# Third-party Library Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn and Related Libraries
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    LeaveOneGroupOut,
    KFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    mutual_info_classif,
    SequentialFeatureSelector,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import mode
import shap

# Local Module Imports
from config import (
    global_path,
    saved_result_path,
    saved_result_path_classification,
    important_vars_path,
)
from utilities import *
from processing import *
from plotting import *


# ask if consider patients with no sympthoms
Input = input("Do you want to consider the reduced df? (y/n)")  # Complete

if Input == "y":
    # Constants and Paths
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_no_symp.csv")  # Reduced
    # GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_red"
    )
else:
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_symp.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_compl"
    )


# Ensure necessary directories exist
os.makedirs(EXPERIMENT_PATH, exist_ok=True)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""

    # file_name = f"classification_reports_{current_datetime}_mrmr.txt"
    # ask if rename the output file
    ans = input("Do you want to rename the output file? (y/n)")
    if ans == "y":
        file_name = input(
            "Insert the name of the output file that follow classification_reports_: "
        )
        file_name = f"classification_reports_{file_name}.txt"
    else:
        file_name = f"classification_reports_ALL.txt"  # {current_datetime}_mrmr.txt"
    sys.stdout = open(os.path.join(EXPERIMENT_PATH, file_name), "w")


# Modify this function
def perform_classification_new(
    clf_model,
    param_grid,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    kf,
    scorer,
    features,
    balancing_technique,
    feature_selection_option,
    results_path,
):
    """
    Perform classification using the specified classifier and evaluate performance.

    Parameters:
        clf_model: Classifier model object.
        param_grid: Parameter grid for grid search.
        X_train (array-like): Features of the training set.
        X_test (array-like): Features of the test set.
        y_train (array-like): Target variable of the training set.
        y_test (array-like): Target variable of the test set.
        kf (KFold): Cross-validation iterator.
        scorer (object): Scorer for model evaluation.
        features (list): List of feature names.
        balancing_technique (str): Balancing technique used.
        feature_selection_option (str): Feature selection option used.
        results_path (str): Path to save classification results.

    Returns:
        best_estimator, best_score_, results_to_save
    """

    # Initialize whether to use the feature selector
    use_selector = False

    # Check if feature selection should be used, assuming `mrmr` is only applicable for certain models
    if feature_selection_option == "mrmr":
        print("Use the selector")
        use_selector = True

    # create a pipeline if mrmr is  used
    if feature_selection_option == "mrmr":
        # Create the pipeline steps based on whether the selector is used
        pipeline_steps = [("imputer", SimpleImputer(strategy="mean"))]

        if use_selector:
            # Configure the sequential feature selector
            selector = SequentialFeatureSelector(
                estimator=clf_model,
                direction="forward",
                scoring=scorer,
                cv=kf,
                n_jobs=-1,
            )
            # Add selector to the pipeline steps only if applicable
            pipeline_steps.append(("selector", selector))

        # Add classifier to the pipeline steps
        pipeline_steps.append(("clf", clf_model))

        # Create the pipeline using the specified steps
        pipeline = Pipeline(pipeline_steps)

        # Map parameter names to the pipeline format
        param_grid = {f"clf__{param}": values for param, values in param_grid.items()}

        # Map original hyperparameter names to pipeline format
        if isinstance(clf_model, RandomForestClassifier):
            new_keys = {
                "n_estimators": "clf__n_estimators",
                "max_depth": "clf__max_depth",
                "min_samples_split": "clf__min_samples_split",
                "min_samples_leaf": "clf__min_samples_leaf",
                "max_features": "clf__max_features",
            }
        elif isinstance(clf_model, XGBClassifier):
            new_keys = {
                "max_depth": "clf__max_depth",
                "n_estimators": "clf__n_estimators",
                "learning_rate": "clf__learning_rate",
                "subsample": "clf__subsample",
                "colsample_bytree": "clf__colsample_bytree",
                "reg_alpha": "clf__reg_alpha",
            }
        elif isinstance(clf_model, SVC):
            new_keys = {
                "C": "clf__C",
                "gamma": "clf__gamma",
                "kernel": "clf__kernel",
            }
        elif isinstance(clf_model, DecisionTreeClassifier):
            new_keys = {
                "max_depth": "clf__max_depth",
                "criterion": "clf__criterion",
                "min_samples_split": "clf__min_samples_split",
                "min_samples_leaf": "clf__min_samples_leaf",
                "max_features": "clf__max_features",
            }
        else:
            new_keys = {k: f"clf__{k}" for k in param_grid.keys()}

        # Update parameter grid to match pipeline configuration if mrmr
        # if feature_selection_option == "mrmr":
        param_grid = {new_keys.get(k, k): v for k, v in param_grid.items()}

    # Initialize GridSearchCV with the pipeline or model
    grid_search = GridSearchCV(
        estimator=pipeline if feature_selection_option == "mrmr" else clf_model,
        param_grid=param_grid,
        cv=kf,  # kf, #TODO PUT IT on kf
        scoring=scorer,
        verbose=1,  # TODO put on 1
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    # Extract results
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_score_ = (
        grid_search.best_score_
    )  # performance metric of the best_estimator on the train set
    cv_results = grid_search.cv_results_

    # Evaluate on the test set
    # y_pred = best_estimator.predict(X_test)
    # conf_matrix = confusion_matrix(y_test, y_pred)

    # Store results
    results_to_save = {
        "best_params": best_params,
        "best_estimator": best_estimator,
        "best_score": best_score_,
        "cv_results": cv_results,
        # "y_pred": y_pred,
        "y_test": y_test,
        # "classification_report": classification_report(
        # y_test, y_pred, output_dict=True
        # ),
        # "confusion_matrix": conf_matrix,
    }

    # Create results directory if needed and save pickle
    os.makedirs(results_path, exist_ok=True)
    results_file_path = os.path.join(
        results_path,
        f"{clf_model.__class__.__name__}_{balancing_technique}_{feature_selection_option}_results.pkl",
    )
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved to {results_file_path}")

    return best_estimator, best_score_, results_to_save


def main():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_output(current_datetime)
    # Load and preprocess data
    df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH, EXPERIMENT_PATH)
    # remove gendna_type from the features

    X, y = define_X_y(df)
    df = df.drop(columns=["gendna_type", "Unnamed: 0"])

    # print the columns
    print("Columns:", df.columns)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        _,
        features,
        kf,
        scorer,
        thr,
        nFeatures,
        num_folds,
    ) = experiment_definition(X, y, df, EXPERIMENT_PATH, mt_DNA_patients)

    print_data_info(
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        df.drop(columns=["subjid"]),
        EXPERIMENT_PATH,
    )

    input("Press Enter to start the classification...")

    print("Starting the classification...")

    classifiers = {  # light
        "XGBClassifier": (
            XGBClassifier(),
            {
                "max_depth": [3, 6],  # Essential depths
                "n_estimators": [50, 150],  # Key numbers of trees
                "learning_rate": [0.1],  # Commonly effective rate
                "subsample": [1.0],  # Full sample
                "colsample_bytree": [0.8],  # Single choice for simplicity
                "reg_alpha": [0],  # No regularization
                "reg_lambda": [1],  # Default value
            },
        ),
    }

    classifiers_final = {
        "XGBClassifier": (
            XGBClassifier(),
            {
                "max_depth": [2, 5, 10, 20],  # Most impactful depths
                "n_estimators": [
                    50,
                    100,
                    200,
                    300,
                ],  # Reduced number of trees
                "learning_rate": [0.01, 0.1, 0.2],  # Wide range learning rates
                "subsample": [0.8, 1.0],  # Essential variations
                "colsample_bytree": [0.8, 1],  # Stable choice
            },
        ),
        "SVM": (
            SVC(),
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Removed extremes
                "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Focus on usable range
                "kernel": ["linear", "rbf"],  # Default kernel
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(),
            {
                "n_estimators": [
                    50,
                    100,
                    200,
                    300,
                ],  # Reduced number of trees
                "max_depth": [2, 5, 10, 20],  # Most impactful depths
                "min_samples_split": [2, 5, 10, 20],  # Regular split thresholds
                "min_samples_leaf": [1, 2, 5, 10],  # Slight variant in leaves
            },
        ),
    }

    # define the settings of the experiment
    balancing_techniques = [
        "no",
        "smote",
        "ada",
    ]

    feature_selection_options = [
        "no",
        "mrmr_ff",  # mrmr with forward feature selection
    ]
    # List to hold results of classifiers
    best_classifiers = []

    # Iterate over each classifier and configuration
    for classifier, (clf_model, param_grid) in classifiers.items():
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        pipeline = Pipeline([("clf", clf_model)])  # Create pipeline to clf

        """# Decrease
        samples = 30
        X_train = X_train[:samples, :]
        X_test = X_test[:samples, :]
        y_train = y_train[:samples]
        y_test = y_test[:samples]
        """

        for feature_selection_option in feature_selection_options:
            (
                X_train_selected,
                X_test_selected,
                param_grid_selected,
                pipeline_selected,
            ) = process_feature_selection(
                clf_model,
                df,
                X_train,
                X_test,
                y_train,
                param_grid,
                pipeline,
                scorer,
                kf,
                feature_selection_option,
                features,
                num_folds,
                nFeatures,
                thr,
            )

            for balancing_technique in balancing_techniques:
                print("------------------------------------------------------------\n")
                print(
                    f"Classifier: {classifier}, Balancing Technique: {balancing_technique}, Feature Selection Option: {feature_selection_option}"
                )

                # Balance data
                X_train_bal, y_train_bal, X_test_bal, y_test_bal = balance_data(
                    X_train_selected,
                    y_train,
                    X_test_selected,
                    y_test,
                    balancing_technique,
                )

                # Perform classification and collect results
                best_estimator, best_score, results = perform_classification_new(
                    clf_model,
                    param_grid_selected,
                    pipeline_selected,
                    X_train_bal,
                    X_test_bal,
                    y_train_bal,
                    y_test_bal,
                    kf,
                    scorer,
                    features,
                    balancing_technique,
                    feature_selection_option,
                    results_path=EXPERIMENT_PATH,
                )

                # Track best classifiers
                best_classifiers.append(
                    (classifier, best_estimator, best_score, results)
                )

                # Print the training performances
                print("Training performances:")
                print("Best score:", best_score)
                print("Best estimator:", best_estimator)
                # print("Results:", results)

                # print end of training
                print("End of training")
                print("------------------------------------------------------------\n")

    results_file_path = os.path.join(
        EXPERIMENT_PATH, f"all_classifiers_{current_datetime}.pkl"
    )
    with open(results_file_path, "wb") as f:
        pickle.dump(best_classifiers, f)
    print(f"Results saved to {results_file_path}")

    # Find and save the best classifier
    best_classifier = max(
        best_classifiers, key=lambda x: x[2]
    )  # Assuming best_score is the 3rd element

    print("********************************************************************\n")
    print("Best Classifier Overall:")
    print(f"Classifier: {best_classifier[0]}")
    print(f"Best Score: {best_classifier[2]}")
    print(f"Best Estimator: {best_classifier[1]}")

    results_file_path = os.path.join(
        EXPERIMENT_PATH, f"best_classifier_{current_datetime}.pkl"
    )
    with open(results_file_path, "wb") as f:
        pickle.dump(best_classifier, f)

    print(f"Results saved to {results_file_path}")
    # print that classification is done
    print("Classification is done and best classifier is saved")


if __name__ == "__main__":
    main()
