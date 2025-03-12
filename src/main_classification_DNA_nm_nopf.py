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
        file_name = f"classification_reports_{current_datetime}_nopf.txt"  # ALL.txt"  # {current_datetime}_mrmr.txt"
    # print the results are saved in the file
    print(f"Results are saved in the file: {file_name}")
    # sys.stdout = open(os.path.join(EXPERIMENT_PATH, file_name), "w")


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

        # if SVM, do a RandomizedSearchCV
    if clf_model.__class__.__name__ == "SVC":
        #print randomized 
        print("Randomized search in progress...")
        grid_search = RandomizedSearchCV(
            estimator=clf_model,
            param_distributions=param_grid,
            cv=kf,  # kf, #TODO PUT IT on kf
            scoring=scorer,
            verbose=1,
            n_jobs=-1,  # TODO put on -1
            return_train_score=True,
            n_iter=10,
        )
        # Initialize GridSearchCV with the pipeline or model
    else:
        grid_search = GridSearchCV(
            estimator=clf_model,
            param_grid=param_grid,
            cv=kf,  # kf, #TODO PUT IT on kf
            scoring=scorer,
            verbose=1,
            n_jobs=-1,  # TODO put on -1
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

    # Create results directory if needed
    # os.makedirs(results_path, exist_ok=True)
    results_file_path_cl = os.path.join(results_path, f"{clf_model.__class__.__name__}")

    # Ensure now that results_file_path_cl is a directory
    # os.makedirs(results_file_path_cl, exist_ok=True)

    # Construct file path, this should be a file, not directory
    results_file_path_complete = os.path.join(
        results_file_path_cl,
        f"{clf_model.__class__.__name__}_{balancing_technique}_{feature_selection_option}_{pf}_{len(feature_set)}_results.pkl",
    )

    # Save results to file
    with open(results_file_path_complete, "wb") as f:
        pickle.dump(results_to_save, f)

    print(f"Results saved to {results_file_path_complete}")

    best_scores = (f1_score_train, accuracy_train)

    return best_estimator, best_score_, best_scores, results_to_save


current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_output(current_datetime)
# Load and preprocess data
df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH, EXPERIMENT_PATH)
# remove gendna_type from the features

# reduce rows of the df
# df = df[:100]

X, y = define_X_y(df)
df = df.drop(columns=["gendna_type", "Unnamed: 0"])

# print the columns
# print("Columns:", df.columns)
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

classifiers_l = {  # light
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": [3],  # Essential depths
            "n_estimators": [50],  # Key numbers of trees
            "learning_rate": [0.5],  # Commonly effective rate
            "subsample": [1.0],  # Full sample
            "colsample_bytree": [0.8],  # Single choice for simplicity
        },
    ),
}

classifiers = {
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
    "SVM": (
        SVC(),
        {
            "C": [0.01, 0.1, 1, 10, 100],  # Removed extremes
            "gamma": [0.01, 0.1, 1, 10, 100],  # Focus on usable range
            "kernel": ["linear", "rbf"],  # Default kernel
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

penalty = ["nopf"]  # TODO CHANGE
# List to hold results of classifiers
best_classifiers = []
# Store all models and configurations
all_scores = []
all_models = []
all_configs = []

# Perform scaling and apply penalty factor if required
X_train_scaled, X_test_scaled = scale_data(X_train, X_test, penalty[0])

# print X_df.columns
# print X_df.columns
print("X_df columns:", X_df.columns)

# convert to df
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)
y_train_df = pd.DataFrame(y_train, columns=["gendna_type"])


# Define the path to the file
file_path = os.path.join(EXPERIMENT_PATH, "feature_sets_mrmr_ff.json")

# Check if the file exists before opening
if os.path.exists(file_path):
    # Load the file if it exists
    with open(file_path, "r") as f:
        feature_sets_mrmr = json.load(f)
    print("Feature sets loaded from file:", feature_sets_mrmr)
else:
    # Create the feature sets if the file doesn't exist
    feature_sets_mrmr = process_feature_selection_mrmr_ff(X_train_scaled_df, y_train_df)
    print("Feature sets created:", feature_sets_mrmr)

    # Save the feature sets to the file as JSON
    with open(file_path, "w") as f:
        json.dump(feature_sets_mrmr, f, indent=4)


# Iterate over each classifier and configuration
for classifier, (clf_model, param_grid) in classifiers.items():
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    # pipeline = Pipeline([("clf", clf_model)])  # Create pipeline to clf

    # Get the class name of the model
    model_name = clf_model.__class__.__name__

    # Get the directory path for this model's results
    model_directory = os.path.join(EXPERIMENT_PATH_RESULTS, model_name)
    # Ensure this directory exists
    os.makedirs(model_directory, exist_ok=True)

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
        """

        # Perform scaling and apply penalty factor if required
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test, pf)

        # print X_df.columns
        # print X_df.columns
        print("X_df columns:", X_df.columns)

        # convert to df
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)
        y_train_df = pd.DataFrame(y_train, columns=["gendna_type"])
        """

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
                feature_sets = feature_sets_mrmr
                # print("Feature sets created:", feature_sets)
            if not feature_sets:
                print(
                    "Warning: No feature sets were generated by the feature selection process."
                )

            # print the length of the feature sets
            # print("Length of feature sets:", len(feature_sets))

            for feature_set in feature_sets:
                # print("Processing feature set:", feature_set)
                # extract
                # print len of feature_set
                # print("Length of feature set:", )
                # print("Columns of X_train_scaled_df:", X_train_scaled_df.columns)
                X_train_subset = X_train_scaled_df[feature_set]
                X_test_subset = X_test_scaled_df[feature_set]

                for balancing_technique in balancing_techniques:
                    # Construct the complete file path for saving results
                    results_filename = f"{model_name}_{balancing_technique}_{feature_selection_option}_{pf}_{len(feature_set)}_results.pkl"
                    results_file_path_complete = os.path.join(
                        model_directory, results_filename
                    )

                    if os.path.exists(results_file_path_complete):
                        # print the name of the file
                        print(f"File already present: {results_filename}")
                        continue  # skip the current iteration
                    else:
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

                        """
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
                            """

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
# find the best model idx using the max f1 score and the max accuracy
best_model_idx = sorted_scores_with_indices[0][0]
print(f"Best model index: {best_model_idx + 1}")
# print the best model
print("Best Model:")
print(all_configs[best_model_idx])
print("Best Model F1-score:", all_scores[best_model_idx][0])
print("Best Model Accuracy:", all_scores[best_model_idx][1])

model_idx = best_model_idx - 1  #
"""(
    int(input("Enter the index number of the model to evaluate on the test set: "))
    - 1
)"""

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

conf_mat = confusion_matrix(y_test, y_pred)
class_labels = ["nDNA", "mtDNA"]
group_names = ["TN", "FP", "FN", "TP"]
group_counts = [f"{value:0.0f}" for value in conf_mat.flatten()]
labels = np.asarray(
    [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
).reshape(2, 2)

sns_plot = sns.heatmap(
    conf_mat, annot=labels, fmt="", cmap="Blues", annot_kws={"size": 18}
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=[0.5, 1.5], labels=class_labels)
plt.yticks(ticks=[0.5, 1.5], labels=class_labels)
figure = plt.gcf()
figure.set_size_inches(7, 6)
plt.savefig(
    os.path.join(EXPERIMENT_PATH, "confusion_matrix_best.png"),
    format="png",
    bbox_inches="tight",
)
plt.close()

# Assuming that mtDNA is the positive class, compute sensitivity and specificity
spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
sens = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
print(f"Sensitivity: {sens:.3f}")
print(f"Specificity: {spec:.3f}")

names = selected_config["feature set"]
print("Feature set:", names)
print("Length of feature set:", len(names))

# try importances otherwise print the error
try:
    importances = selected_model.feature_importances_
    names = selected_config["feature set"]  # X_df.columns
    model_type = selected_model.__class__.__name__
    plot_top_feature_importance(
        importances,
        names,
        model_type,
        save_path=EXPERIMENT_PATH,
        top_n=10,
    )
except Exception as e:
    print("Error in feature importance:", e)

# print end of the script
print("End of the script")
