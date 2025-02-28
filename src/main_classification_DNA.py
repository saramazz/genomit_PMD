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

from sklearn.utils.class_weight import compute_class_weight

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


EXPERIMENT_PATH_RESULTS = os.path.join(EXPERIMENT_PATH, "results")

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_PATH, exist_ok=True)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""

    # file_name = f"classification_reports_{current_datetime}_mrmr.txt"
    # ask if rename the output file
    ans = input(
        "Do you want to rename the output file? (y/n) Default is classification_reports_ALL.txt"
    )
    if ans == "y":
        file_name = input(
            "Insert the name of the output file that follow classification_reports_: "
        )
        file_name = f"classification_reports_{file_name}.txt"
    else:
        file_name = f"classification_reports_ALL.txt"  # {current_datetime}_mrmr.txt"
    sys.stdout = open(os.path.join(EXPERIMENT_PATH, file_name), "w")


def perform_classification(
    clf_model,
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
    kf,
    scorer,
    features,
    feature_set,
    balancing_technique,
    pf,
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

    # Initialize GridSearchCV with the pipeline or model
    grid_search = GridSearchCV(
        estimator=clf_model,
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
    y_pred_train = best_estimator.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    f1_score_train = f1_score(y_train, y_pred_train)
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    print(conf_matrix_train)
    print(f"Training Accuracy: {accuracy_train:.3f}")
    print(f"F1-score: {f1_score_train:.3f}")

    # Store results
    results_to_save = {
        "best_params": best_params,
        "best_estimator": best_estimator,
        "best_score": best_score_,
        "cv_results": cv_results,
        "y_test": y_test,
        "feature set": feature_set,
        "features": len(feature_set),
        "sampling": balancing_technique,
        "model": clf_model.__class__.__name__,
        "hyperparameters": grid_search.best_params_,
        "conf_matrix": conf_matrix_train,
        "accuracy": accuracy_train,
        "f1_score": f1_score_train,
    }

    # Create results directory if needed and save pickle
    os.makedirs(results_path, exist_ok=True)
    results_file_path = os.path.join(
        results_path,
        f"{clf_model.__class__.__name__}_{balancing_technique}_{feature_selection_option}_{pf}_{len(feature_set)}_results.pkl",
    )
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved to {results_file_path}")

    best_scores = (f1_score_train, accuracy_train)

    return best_estimator, best_score_, best_scores, results_to_save


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

    X_df = df.drop(columns=["subjid", "test"])

    print_data_info(
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        X_df,  # df.drop(columns=["subjid"]),
        EXPERIMENT_PATH,
    )

    # input("Press Enter to start the classification...")

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
        "mrmr_ff",  # mrmr with forward feature selection
        "no",
    ]

    penalty = ["nopf", "pf"]
    # List to hold results of classifiers
    best_classifiers = []
    # Store all models and configurations
    all_scores = []
    all_models = []
    all_configs = []

    # Iterate over each classifier and configuration
    for classifier, (clf_model, param_grid) in classifiers.items():
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        #pipeline = Pipeline([("clf", clf_model)])  # Create pipeline to clf

        """# Decrease
        samples = 30
        X_train = X_train[:samples, :]
        X_test = X_test[:samples, :]
        y_train = y_train[:samples]
        y_test = y_test[:samples]
        """

        for pf in penalty:
            print("Penalty factor:", pf)

            # Print before scaling
            # print("Before scaling:")
            # print("X_train sample:\n", X_train[:5])  # Print first 5 rows
            # print("X_test sample:\n", X_test[:5])

            # Perform scaling and apply penalty factor if required
            X_train_scaled, X_test_scaled = scale_data(X_train, X_test, pf)

            # print X_df.columns
            # print X_df.columns
            print("X_df columns:", X_df.columns)

            # convert to df
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)
            y_train_df = pd.DataFrame(y_train, columns=["gendna_type"])

            # Print after scaling
            # print("\nAfter scaling:")
            # print("X_train_scaled sample:\n", X_train_scaled[:5])
            # print("X_test_scaled sample:\n", X_test_scaled[:5])

            # input("Press Enter to continue...")

            for feature_selection_option in feature_selection_options:
                print("Processing feature selection option:", feature_selection_option)
                if feature_selection_option == "no":
                    print("No feature selection, using all features.")
                    feature_sets = [X_df.columns]
                else:
                    # Perform feature selection with mRMR and forward feature selection
                    print(
                        "Performing feature selection with mRMR and forward feature selection."
                    )
                    feature_sets = process_feature_selection_mrmr_ff(
                        X_train_scaled_df, y_train_df
                    )
                    print("Feature sets created:", feature_sets)
                if not feature_sets:
                    print(
                        "Warning: No feature sets were generated by the feature selection process."
                    )

                for feature_set in feature_sets:
                    print("Processing feature set:", feature_set)
                    #print len of feature_set
                    print("Length of feature set:", len(feature_set))
                    X_train_subset = X_train_scaled_df[feature_set]
                    X_test_subset = X_test_scaled_df[feature_set]



                    for balancing_technique in balancing_techniques:
                        print(
                            "------------------------------------------------------------\n"
                        )
                        print(
                            f"Classifier: {classifier}, Balancing Technique: {balancing_technique}, Feature Selection Option: {feature_selection_option}, Penalty Factor: {pf}"
                        )

                        # Balance data
                        X_train_bal, y_train_bal, X_test_bal, y_test_bal = balance_data(
                            X_train_subset,
                            y_train,
                            X_test_subset,
                            y_test,
                            balancing_technique,
                        )

                        # Print the dimension of the data
                        print("X_train_bal shape:", X_train_bal.shape)
                        print("X_test_bal shape:", X_test_bal.shape)

                        '''
                        # Compute class weights
                        class_weights = compute_class_weight(
                            "balanced", classes=np.unique(y_train_bal), y=y_train_bal
                        )
                        class_weight_dict = dict(
                            zip(np.unique(y_train_bal), class_weights)
                        )

                        # Add the class_weight_dict to the model if it is not SVM
                        if classifier != "SVM":
                            # Set the class_weight parameter if the classifier supports it
                            clf_model.set_params(class_weight=class_weight_dict)
                            '''

                        # Perform classification and collect results
                        best_estimator, best_score, best_scores, results = (
                            perform_classification(
                                clf_model,
                                param_grid,
                                X_train_bal,
                                X_test_bal,
                                y_train_bal,
                                y_test_bal,
                                kf,
                                scorer,
                                features,
                                feature_set,
                                balancing_technique,
                                pf,
                                feature_selection_option,
                                results_path=EXPERIMENT_PATH_RESULTS,
                            )
                        )

                        # Save all models and configurations
                        all_scores.append(best_scores)
                        all_models.append(
                            best_estimator
                        )  # grid_search.best_estimator_)
                        all_configs.append(results)

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
                        print(
                            "------------------------------------------------------------\n"
                        )

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

    # Sort the scores in descending order based on the F1-score (first value in tuple), keeping corresponding accuracy (second value)
    sorted_scores_with_indices = sorted(
        enumerate(all_scores), key=lambda x: x[1][0], reverse=True
    )

    # Ask user to choose the best model for testing
    print("\nChoose the best model configuration for the test set:")
    for idx, (original_idx, (f1_training, accuracy_training)) in enumerate(
        sorted_scores_with_indices
    ):
        print(
            f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}"
        )

    model_idx = (
        int(input("Enter the index number of the model to evaluate on the test set: "))
        - 1
    )

    # Evaluate selected model
    selected_model = all_models[model_idx]
    selected_config = all_configs[model_idx]
    print("\nEvaluating Best Model on Test Set:")

    y_pred = selected_model.predict(X_test_scaled_df[selected_config["feature set"]])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["nDNA", "mtDNA"],
        yticklabels=["nDNA", "mtDNA"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Selected Model")
    # save the confusion matrix
    plt.savefig(os.path.join(EXPERIMENT_PATH, "confusion_matrix_best.png"))

    # Assuming that mtDNA is the positive class, compute sensitivity and specificity
    spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    sens = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")

    # print end of the script
    print("End of the script")


if __name__ == "__main__":
    main()
