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
import joblib
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

# Constants and Paths
GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
EXPERIMENT_PATH = os.path.join(
    saved_result_path_classification, "experiments_all_models"
)

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_PATH, exist_ok=True)


def load_and_prepare_data():
    """Load and prepare the DataFrame for classification."""
    if not os.path.exists(GLOBAL_DF_PATH):
        raise FileNotFoundError(f"File not found: {GLOBAL_DF_PATH}")

    df = pd.read_csv(GLOBAL_DF_PATH)

    # Display initial DataFrame information
    nRow, nCol = df.shape
    print(f'The DataFrame "df_preprocessed" contains {nRow} rows and {nCol} columns.')

    # print("Columns:", df.columns)

    # Preprocess target column, handle NaNs, and print DataFrame
    df, df_not_numerical = process_gendna_column(df)
    df = fill_missing_values(df)

    # Fill the remaining missing values with 998
    df = df.fillna(998)

    # Check if there are any remaining missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(
            f"Warning: There are still {missing_count} missing values in the DataFrame."
        )
    else:
        print("All missing values have been successfully filled.")

    return df


def feature_selection(df):
    """Select and drop unnecessary features from the DataFrame."""
    df_vars = pd.read_excel(important_vars_path)
    column_name = "consider for mtDNA vs nDNA classification?"
    columns_to_drop = list(df_vars.loc[df_vars[column_name] == "N", "variable"])

    # Additional columns to drop
    additional_columns = [
        "Hospital",
        "nDNA",
        "mtDNA",
        "gendna_type",
        "epiphen",
        "sll",
        "clindiag__decod",
        "encephalopathy",
    ]
    additional_columns += [
        col for col in df.columns if "pimgtype" in col or "psterm" in col
    ]

    columns_to_drop += additional_columns
    # print("Columns to drop:", columns_to_drop)

    X, y = define_X_y(df, columns_to_drop)
    return X, y


def print_data_info(X_train, X_test, y_train, y_test, features, df):
    """Prints information about the datasets and features."""
    # print the experiment path
    print("Experiment path:", EXPERIMENT_PATH)

    print("Dataset for classification shape:", df.shape)

    print("Dimension of X_train:", X_train.shape)
    print("Dimension of X_test:", X_test.shape)
    print("Dimension of y_train:", y_train.shape)
    print("Dimension of y_test:", y_test.shape)
    print("Features names:", features)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""

    # file_name = f"classification_reports_{current_datetime}_mrmr.txt"
    file_name = f"classification_reports_mrmr_XGB_new.txt"  # {current_datetime}_mrmr.txt"
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

    """

     # Set up feature selection and pipeline
    if feature_selection_option == "mrmr":
        # Configure the SequentialFeatureSelector
        selector = SequentialFeatureSelector(
            clf_model, direction="forward", scoring=scorer, cv=kf, n_jobs=-1
        )
        
        # Configure the pipeline with an imputer, selector, and classifier
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("selector", selector),
                ("clf", clf_model),
            ]
        )

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

        # Update parameter grid to match pipeline configuration
        param_grid = {new_keys.get(k, k): v for k, v in param_grid.items()} 

    # Initialize GridSearchCV with the pipeline or model
    grid_search = GridSearchCV(
        estimator=pipeline if feature_selection_option == "mrmr" else clf_model,
        param_grid=param_grid,
        cv=kf,
        scoring=scorer,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
    )
    """


    # Initialize whether to use the feature selector
    use_selector = False
    '''

    # Check if feature selection should be used, assuming `mrmr` is only applicable for certain models
    if feature_selection_option == "mrmr" and isinstance(clf_model, LightGBMClassifier):
        print("Use the selector")
        use_selector = True
    '''

    # Create the pipeline steps based on whether the selector is used
    pipeline_steps = [("imputer", SimpleImputer(strategy="mean"))]

    if use_selector:
        # Configure the sequential feature selector
        selector = SequentialFeatureSelector(
            estimator=clf_model, direction="forward", scoring=scorer, cv=kf, n_jobs=-1
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

    # Update parameter grid to match pipeline configuration
    param_grid = {new_keys.get(k, k): v for k, v in param_grid.items()}

    # Initialize GridSearchCV with the pipeline or model
    grid_search = GridSearchCV(
        estimator=pipeline if feature_selection_option == "mrmr" else clf_model,
        param_grid=param_grid,
        cv=kf,  # kf, #TODO PUT IT on kf
        scoring=scorer,
        verbose=2,  # TODO put on 1
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
    df = load_and_prepare_data()
    X, y = feature_selection(df)
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
    ) = experiment_definition(X, y, df, EXPERIMENT_PATH)

    print_data_info(
        X_train, X_test, y_train, y_test, features, df.drop(columns=["subjid"])
    )

    # input("Press Enter to start the classification...")

    print("Starting the classification...")

    """
    classifiers = {
        "DecisionTreeClassifier": (
            DecisionTreeClassifier(),
            {
                "max_depth": [None, 10, 20],  # Key depths including no limit
                "min_samples_split": [2, 10],  # Sparing values
                "min_samples_leaf": [1, 4],  # Only two options to compare leaf sizes
                "criterion": ["gini", "entropy"],  # Two main impurity measures
                "max_features": [None, "sqrt"],  # Two useful options
                "splitter": ["best"],  # Default and commonly useful setting
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 300],  # Key sizes of forests
                "max_depth": [None, 20],  # Either unconstrained or some constraint
                "min_samples_split": [2, 10],  # Common divisors
                "min_samples_leaf": [1, 3],  # Two simple leaves
                "max_features": [None, "sqrt"],  # Reduce features considered per split
                "bootstrap": [True],  # Most typical setting
                "criterion": ["gini"],  # Focus on gini
            },
        ),
        "SVM": (
            SVC(),
            {
                "C": [0.1, 1, 10],  # Simplified C range
                "gamma": [0.01, 0.1],  # Simplified gamma range
                "kernel": ["rbf"],  # Broadly useful and performant kernel
            },
        ),
    }
    """


    classifiers = {
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": [3, 6, 9],  # Reduced range
            "n_estimators": [50, 150, 250],  # Fewer values
            "learning_rate": [0.01, 0.1],  # Key learning rates
            "subsample": [0.8, 1.0],  # Essential subsample rates
            "colsample_bytree": [0.8, 1.0],  # Two core options
            "reg_alpha": [0, 0.1],  # Reduced scrutiny on regularization
            "reg_lambda": [1, 2]  # Just two prominent values
        },
    ),
    }

    # define the settings of the experiment
    balancing_techniques = [
        "no",
        "over",
        "under",
    ]
    # feature_selection_options = ["no", "pca", "select_from_model", "rfe"] #,MRMR
    feature_selection_options = ["mrmr"]  # ,MRMR

    # List to hold results of classifiers
    best_classifiers = []

    # Iterate over each classifier and configuration
    for classifier, (clf_model, param_grid) in classifiers.items():
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        pipeline = Pipeline([("clf", clf_model)])  # Create pipeline to clf

        # Create an imputer object with 'constant' strategy and your specified fill value
        imputer = SimpleImputer(strategy="constant", fill_value=998)

        # Fit the imputer on the training data and transform it
        X_train = imputer.fit_transform(X_train)

        # Transform the test data with the same imputer
        X_test = imputer.transform(X_test)

        """
        #Decrease
        samples=30
        X_train = X_train[ :samples, :]
        X_test = X_test[:samples,:]
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
"""
# Define the classifiers and their parameter grids

classifiers = {
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": range(2),
            "n_estimators": range(5),
            "learning_rate": [0.2],
        },
    ),
}

# define the settings of the experiment
balancing_techniques = [
    "under",
]
feature_selection_options = ["no"]

'''

classifiers = {
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": range(2, 22, 4),
            "n_estimators": range(50, 500, 50),
            "learning_rate": [0.01, 0.1, 0.2],
        },
    ),
    "DecisionTreeClassifier": (
        DecisionTreeClassifier(),
        {
            "max_depth": range(2, 30, 2),
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2", None],
        },
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 300, 500],  # Increased to 3 values
            "max_depth": [None, 10, 20, 30],  # 4 values
            "min_samples_split": [2, 5, 10],  # 3 values
            "min_samples_leaf": [1, 2, 4],  # 3 values
            "max_features": ["auto", "sqrt"],  # 2 values
            "bootstrap": [True, False],  # 2 values
            "criterion": ["gini", "entropy"],  # 2 values
        },
    ),
    "SVM": (
        SVC(),
        {
            "C": [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
            "kernel": ["rbf"],
        },
    ),
}

# define the settings of the experiment
balancing_techniques = [
    "no",
    "over",
    "under",
]
feature_selection_options = ["no", "pca", "mrmr", "select_from_model", "rfe"]

"""


"""
for classifier, (clf_model, param_grid) in classifiers.items():
    print("********************************************************************\n")
    pipeline = Pipeline([("clf", clf_model)])  # create pipeline to clf

    for feature_selection_option in feature_selection_options:
        # Apply feature selection option to the data if necessary
        # Process feature selection
        X_train_selected, X_test_selected, param_grid_selected, pipeline_selected = (
            process_feature_selection(
                clf_model,
                df#X_df,
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
        )

        for balancing_technique in balancing_techniques:
            # print the current classification settings in one line only
            print("------------------------------------------------------------\n")
            print(
                f"Classifier: {classifier}, Balancing Technique: {balancing_technique}, Feature Selection Option: {feature_selection_option}"
            )

            # Apply balancing technique to the data if necessary
            X_train_bal, y_train_bal, X_test_bal, y_test_bal = balance_data(
                X_train_selected, y_train, X_test_selected, y_test, balancing_technique
            )

            # finally, perform the classification
            perform_classification(
                clf_model,
                param_grid,
                pipeline_selected,
                df, #X_df
                X_train_bal,
                X_test_bal,
                y_train_bal,
                y_test_bal,
                kf,
                scorer,
                features,
                balancing_technique,
                feature_selection_option,
                saved_result_path_classification_exp,
            )


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__



# Read the text file
with open(os.path.join(saved_result_path_classification, file_name), 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
classifiers = []
balancing_techniques = []
feature_selection_options = []
weighted_avg_f1_scores = []
accuracies = []
f1_score_mt = []
f1_score_wt = []

# Parse the text file and extract data
for line in lines:
    if line.startswith("Classifier"):
        classifiers.append(line.split(":")[1].strip().split(',')[0].strip())
        balancing_techniques.append(line.split(":")[2].strip().split(',')[0].strip())
        feature_selection_options.append(line.split(":")[3].strip().split(',')[0].strip())
    elif line.startswith("weighted avg"):
        weighted_avg_f1_scores.append(float(line.split()[3]))
        accuracies.append(float(line.split()[2]))
    elif line.startswith("f1-score_mt"):
        f1_score_mt.append(float(line.split()[1]))
        f1_score_wt.append(float(line.split()[3]))

# Create a DataFrame
df = pd.DataFrame({
    'Classifier': classifiers,
    'Balancing Technique': balancing_techniques,
    'Feature Selection Option': feature_selection_options,
    'Weighted Avg F1-Score': weighted_avg_f1_scores,
    'Accuracy': accuracies,
    'F1-Score MT': f1_score_mt,
    'F1-Score WT': f1_score_wt
})

print("Classification results:")
print(df)

# Save the DataFrame to an Excel file
df.to_excel('classification_output.xlsx', index=False)
"""
