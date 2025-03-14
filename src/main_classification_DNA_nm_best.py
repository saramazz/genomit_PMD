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
    file_name = f"classification_reports_best_pf.txt"  # ALL.txt"  # {current_datetime}_mrmr.txt"
    # print the results are saved in the file
    print(f"Results are saved in the file: {file_name}")
    sys.stdout = open(os.path.join(EXPERIMENT_PATH, file_name), "w")


"""
Load all the scores and models from the results path
"""

all_scores = []
all_models = []
all_configs = []

# create the all scores variables reading the files in the results path
for root, dirs, files in os.walk(EXPERIMENT_PATH_RESULTS):
    for file in files:
        # if file do not have nopf in the name
        if "_pf_" in file:  # TODO check is only pf
            if file.endswith(".pkl"):
                with open(os.path.join(root, file), "rb") as f:
                    print(f"Reading file: {file}")
                    # try to read otherwise print the error
                    try:
                        results = pickle.load(f)
                        all_scores.append((results["f1_score"], results["accuracy"]))
                        all_models.append(results["best_estimator"])
                        all_configs.append(results)

                    except Exception as e:
                        print("Error in reading file:", e)
                        continue
                    # print the name of the file


"""
Load the X_test_scaled_df and y_test from the saved path
"""


current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_output(current_datetime)
# Load and preprocess data
df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH, EXPERIMENT_PATH)

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
penalty = ["pf"]  # nopf

# Perform scaling and apply penalty factor if required
X_train_scaled, X_test_scaled = scale_data(X_train, X_test, penalty[0])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)

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
        f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}, Model = {all_configs[original_idx]['model']}"
    )


# Find the best model index using the max F1 score and max accuracy
best_model_idx = sorted_scores_with_indices[0][0]

# Print the index of the best model (zero-based)
print(f"Best model index: {best_model_idx}")

# Retrieve the results dictionary for the best model
best_model_results = all_configs[best_model_idx]

# Print only the desired keys and their corresponding values
keys_to_print = [
    # name classifier
    "model",
    "best_params",
    "best_estimator",
    "best_score",
    "feature set",
    "features",
    "sampling",
    "model",
    "hyperparameters",
    "conf_matrix",
]

print("Best Model Details:")
for key in keys_to_print:
    print(f"{key}: {best_model_results[key]}")

# Print additional results for the best model
print("Best Model F1-score:", all_scores[best_model_idx][0])
print("Best Model Accuracy:", all_scores[best_model_idx][1])

# Evaluate the selected model
selected_model = all_models[best_model_idx]
selected_config = all_configs[best_model_idx]
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

# print confusion matrix of the test set
print("Confusion Matrix of the test set:")
print(conf_mat)
print("Classification Report of the test set:")
print(classification_report(y_test, y_pred, target_names=class_labels))


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


# Find the best model index using the max F1 score and max accuracy
best_model_idx = 422#153 rid # sorted_scores_with_indices[1][0] #second

# Print the index of the best model (zero-based)
print(f"Best model index: {best_model_idx}")

# Retrieve the results dictionary for the best model
best_model_results = all_configs[best_model_idx]

# Print only the desired keys and their corresponding values
keys_to_print = [
    # name classifier
    "model",
    "best_params",
    "best_estimator",
    "best_score",
    "feature set",
    "features",
    "sampling",
    "model",
    "hyperparameters",
    "conf_matrix",
]

print("Best Model Details:")
for key in keys_to_print:
    print(f"{key}: {best_model_results[key]}")

# Print additional results for the best model
print("Best Model F1-score:", all_scores[best_model_idx][0])
print("Best Model Accuracy:", all_scores[best_model_idx][1])

# Evaluate the selected model
selected_model = all_models[best_model_idx]
selected_config = all_configs[best_model_idx]
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
    os.path.join(EXPERIMENT_PATH, "confusion_matrix_best_2.png"),
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

# print confusion matrix of the test set
print("Confusion Matrix of the test set:")
print(conf_mat)
print("Classification Report of the test set:")
print(classification_report(y_test, y_pred, target_names=class_labels))


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

# Find the best model index using the max F1 score and max accuracy
best_model_idx = 161# Rd204  # sorted_scores_with_indices[1][0] #second

# Print the index of the best model (zero-based)
print(f"Best model index: {best_model_idx}")

# Retrieve the results dictionary for the best model
best_model_results = all_configs[best_model_idx]

# Print only the desired keys and their corresponding values
keys_to_print = [
    # name classifier
    "model",
    "best_params",
    "best_estimator",
    "best_score",
    "feature set",
    "features",
    "sampling",
    "model",
    "hyperparameters",
    "conf_matrix",
]

print("Best Model Details:")
for key in keys_to_print:
    print(f"{key}: {best_model_results[key]}")

# Print additional results for the best model
print("Best Model F1-score:", all_scores[best_model_idx][0])
print("Best Model Accuracy:", all_scores[best_model_idx][1])

# Evaluate the selected model
selected_model = all_models[best_model_idx]
selected_config = all_configs[best_model_idx]
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

# print confusion matrix of the test set
print("Confusion Matrix of the test set:")
print(conf_mat)
print("Classification Report of the test set:")
print(classification_report(y_test, y_pred, target_names=class_labels))



# print end of the script
print("End of the script")
