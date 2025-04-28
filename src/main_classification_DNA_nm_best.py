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
                        all_scores.append(
                            (
                                results["f1_score"],
                                results["accuracy"],
                                results["best_score"],
                            )
                        )
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

"""
# Sort the scores in descending order based on the F1-score (first value in tuple), keeping corresponding accuracy (second value)
sorted_scores_with_indices = sorted(
    enumerate(all_scores), key=lambda x: x[1][0], reverse=True
)
"""

sorted_scores_with_indices = sorted(
    enumerate(all_scores), key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True
)

"""
# Ask user to choose the best model for testing
print("\nChoose the best model configuration for the test set:")
for idx, (original_idx, (f1_training, accuracy_training,best_score)) in enumerate(
    sorted_scores_with_indices
):
    print(
        f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}, Model = {all_configs[original_idx]['model']}, best_score = {best_score:.3f}"
    )
    # print also
    keys_to_print = [
        # name classifier
        "model",
        "best_params",
        "best_score",
        "feature set",
        "features",
        "sampling",
        "model",
    ]
    print("Model details: ")
    for key in keys_to_print:
        print(f"{key}: {all_configs[original_idx][key]}")
    print("\n")
"""

# Assuming you have a list of models with their metrics and configurations
print("\nChoose the best model configuration for the test set:")

# Define a list to keep track of suitable models with complexity considerations
suitable_models = []

for idx, (original_idx, (f1_training, accuracy_training, best_score)) in enumerate(
    sorted_scores_with_indices
):
    model_config = all_configs[original_idx]

    # Check complexity parameters and append suitable models
    complexity_params = {
        "num_trees": model_config.get("best_params", {}).get("n_estimators", 0),
        "max_depth": model_config.get("best_params", {}).get("max_depth", 0),
        "num_leaves": model_config.get("best_params", {}).get("num_leaves", 0),
        "nodes": model_config.get("best_params", {}).get("layers", 0),  # for DL models
    }

    print(
        f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}, Model = {model_config['model']}, best_score = {model_config['best_score']:.3f}, complexity_params = {complexity_params}"
    )

    keys_to_print = [
        "model",
        "best_params",
        "best_score",
        "feature set",
        "features",
        "sampling",
    ]

    # Complexity check: prefer models with fewer trees and lower depth
    if complexity_params["num_trees"] < 30 and complexity_params["max_depth"] < 5:
        suitable_models.append(
            (original_idx, f1_training, accuracy_training, model_config)
        )

# print the number and details of the suitable models
print(
    f"\nNumber of suitable models based on complexity criteria: {len(suitable_models)}"
)
"""
for idx, (model_idx, f1_training, accuracy_training, model_config) in enumerate(
    suitable_models
):  
    print(
        f"Model {idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}, Model = {model_config['model']}, best_score = {model_config['best_score']:.3f}"
    )
    keys_to_print = [
        "model",
        "best_params",
        "best_score",
        "feature set",
        "features",
        "sampling",
    ]
    print("Model details: ")
    for key in keys_to_print:
        print(f"{key}: {model_config[key]}")
    print("\n")
"""


"""
# If multiple suitable models exist, choose the best one based on F1-score or Accuracy
if suitable_models:
    best_model_info = max(
        suitable_models, key=lambda x: x[1]
    )  # Select the model with the highest F1 score

    best_model_idx = best_model_info[0]
    print(
        f"\nSuitable Selected Model: {all_configs[best_model_idx]['model']} with F1-score: {best_model_info[1]:.3f} and Accuracy: {best_model_info[2]:.3f}"
    )

    # Retrieve the best model results
    best_model_results = all_configs[best_model_idx]

    # Evaluate the selected model on the test set
    selected_model = best_model_results[
        "best_estimator"
    ]  # Assuming best_estimator is saved
    selected_config = best_model_results

    print("\nEvaluating Best Model on Test Set:")
    y_pred = selected_model.predict(X_test)  # Use the proper test set defined elsewhere

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
    class_labels = ["Class 0", "Class 1"]  # Adjust class labels as necessary
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
        os.path.join(EXPERIMENT_PATH, "confusion_matrix_best_suit.png"),
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    # Assuming that Class 1 is positive
    spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    sens = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    print(f"Sensitivity: {sens:.3f}")
    print(f"Specificity: {spec:.3f}")

    # Print feature set details
    names = selected_config["feature set"]
    print("Feature set:", names)
    print("Length of feature set:", len(names))

    # Print confusion matrix and classification report
    print("Confusion Matrix of the test set:")
    print(conf_mat)
    print("Classification Report of the test set:")
    print(classification_report(y_test, y_pred, target_names=class_labels))
else:
    print("No suitable models found based on the defined complexity criteria.")
"""


"""
PRINT DETAILED INFO
"""

"""
BEST CLF 1
"""
# Find the best model index using the max F1 score and max accuracy
best_model_idx = 107# sorted_scores_with_indices[0][0]


# Print the index of the best model (zero-based)
print(f"Best model index 0,0: {best_model_idx}")



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
print("\nEvaluating Best Model 1 on Test Set:")

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
    os.path.join(EXPERIMENT_PATH, "confusion_matrix_best1.png"),
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
    model_type = f"{selected_model.__class__.__name__}_1"
    plot_top_feature_importance(
        importances,
        names,
        model_type,
        save_path=EXPERIMENT_PATH,
        top_n=10,
    )
except Exception as e:
    print("Error in feature importance:", e)

    '''

"""
BEST CLF 2
"""

# Find the best model index using the max F1 score and max accuracy
best_model_idx = 422  # red291  # 422#153 rid # sorted_scores_with_indices[1][0] #second

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
print("\nEvaluating Best Model 2 on Test Set:")

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
    model_type = f"{selected_model.__class__.__name__}_2"
    plot_top_feature_importance(
        importances,
        names,
        model_type,
        save_path=EXPERIMENT_PATH,
        top_n=10,
    )
except Exception as e:
    print("Error in feature importance:", e)


"""
BEST CLF 3
"""

# Find the best model index using the max F1 score and max accuracy
best_model_idx = (
    97  # red107  # old comp161# Rd204  # sorted_scores_with_indices[1][0] #second
)

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
print("\nEvaluating Best Model 3 on Test Set:")

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
    os.path.join(EXPERIMENT_PATH, "confusion_matrix_best_3.png"),
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
    model_type = f"{selected_model.__class__.__name__}_3"
    plot_top_feature_importance(
        importances,
        names,
        model_type,
        save_path=EXPERIMENT_PATH,
        top_n=10,
    )
except Exception as e:
    print("Error in feature importance:", e)
    '''

"""
MISCLASSIFICATION FN,FP ANALYSIS  
"""

results = best_model_results


# identify the test subjects in the df raw
df_raw = pd.read_csv(GLOBAL_DF_PATH)
print("df_raw shape:", df_raw.shape)
#print head of df_raw
print("df_raw head:", df_raw.head())
#print columns of df_raw
print("df_raw columns:", df_raw.columns)
#input("Press Enter to continue...")

#test_subjects are the subjid whose test column is 1
test_subjects = df_raw[df_raw["test"] == 1]["subjid"].unique()
#print dimension of the test subjects
print("Test subjects shape:", test_subjects.shape)
df_raw_test = df_raw[df_raw["subjid"].isin(test_subjects)]


print("Results:")
print(results)

#y_pred = results["y_pred"]
#y_test = results["y_test"]

# Identify false positives (FP) and false negatives (FN)
fp_indices = (y_test == 0) & (y_pred == 1)  # mt 0 as nDNA their gen is 5,7
fn_indices = (y_test == 1) & (y_pred == 0)  # nDNA 1 as mt, their gen is 4,6,8

# Get the subject IDs for FP and FN cases
fp_subjids = df_raw_test.loc[fp_indices, "subjid"]
fn_subjids = df_raw_test.loc[fn_indices, "subjid"]

# create a df with the df_raw_test and the predictions and the fn and fp
df_raw_test["predictions"] = y_pred
df_raw_test["fp"] = fp_indices
df_raw_test["fn"] = fn_indices

# print shape of df_test
print("Added to df_test the FP, FN and y_pred :", df_raw_test.shape)

"""
sobstitute HPO code with the name of the symptom
"""

df_test = df_raw_test

symptoms_mapping_path = os.path.join(
    saved_result_path, "mapping", "psterm_modify_association_Besta.xlsx"
)
# Identify columns containing 'symp_on' in their names
columns_with_symptoms = [col for col in df_test.columns if "symp_on" in col]
# print("Columns with symptoms:", columns_with_symptoms)

# Load symptoms mapping file
mapping_symptoms = pd.read_excel(symptoms_mapping_path)

# Clean the 'psterm__decod' column by removing 'HP:'
mapping_symptoms["psterm__decod"] = mapping_symptoms["psterm__decod"].str.replace(
    "HP:", ""
)
import re

# Define a function to extract text between single quotes
def extract_text(text):
    match = re.search(r"'([^']*)'", text)
    return match.group(1) if match else None

# Apply the function to the 'associated_psterm__modify' column
mapping_symptoms["associated_psterm__modify"] = mapping_symptoms[
    "associated_psterm__modify"
].apply(extract_text)
# print("Mapping symptoms:\n", mapping_symptoms)

# Reset index of df_test to ensure unique index values
df_test.reset_index(drop=True, inplace=True)

# Print the columns_with_symptoms of the df_test
# print("Test subjects DataFrame with symptoms codes:\n", df_test[columns_with_symptoms])

# Substitute the symptom codes with names in df_test using the mapping file
symptoms_dict = mapping_symptoms.set_index("psterm__decod")[
    "associated_psterm__modify"
].to_dict()
for col in columns_with_symptoms:
    df_test[col] = df_test[col].map(symptoms_dict)

# Display the modified DataFrame to verify the changes
# print("Modified df_test with symptom names:\n", df_test[columns_with_symptoms])
print("sobsituted HPO code with the name of the symptom in df_test")

"""
add other columns from the original df
"""

# sobstitute using this mapping in the column clindiag__decod
clindiag_mapping = {
    "C01": "MELAS",
    "B01": "CPEO",
    "A02": "ADOA",
    "A01": "LHON",
    "C04": "Leigh syndrome",
    "C19": "Encephalopathy",
    "B02": "CPEO plus",
    "C03": "MERRF",
    "B03": "MiMy (without PEO)",
    "E": "unspecified mitochondrial disorder",
    "C06": "Kearns-Sayre-Syndrome (KSS)",
    "C05": "NARP",
    "C18": "Encephalomyopathy",
    "C02": "MIDD",
    "C17": "Other mitochondrial multisystem disorder",
    "C07": "SANDO/MIRAS/SCAE",
    "F": "asymptomatic mutation carrier",
    "D01": "Isolated mitochondrial Cardiomyopathy",
    "A03": "other MON",
    "C08": "MNGIE",
    "C16": "LBSL",
    "C": "Mitochondrial Multisystem Disorders",
    "C09": "Pearson syndrome",
    "C12": "Wolfram-Syndrome (DIDMOAD-Syndrome)",
    "D05": "Other mitochondrial mono-organ disorder",
}

#add the clindiag__decod column to the df_test from the os.path.join(saved_result_path, "df", "df_symp.csv")
my_df_path=os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
my_df = pd.read_csv(my_df_path)
#add the clindiag__decod only for the common subjid
df_test = pd.merge(df_test, my_df[["subjid","clindiag__decod"]], on="subjid", how="left")


df_test["clindiag__decod"] = df_test["clindiag__decod"].map(clindiag_mapping)

# print("clin diag decod updated", df_test["clindiag__decod"])

# print("Columns added to df_test:\n", df_test[columns_to_add])
# print("size of df_test", df_test.shape)
print("clindiag__decod mapping applied to df_test")

# Definizione della nuova mappatura per pssev
pssev_mapping = {
    "HP:0012827": "Borderline",
    "HP:0012825": "Mild",
    "HP:0012826": "Moderate",
    "HP:0012829": "Profound",
    "HP:0012828": "Severe",
}

# print("columns to be renamed", df_test.columns)
# Aggiornamento della colonna pssev utilizzando la nuova mappatura
for index, row in df_test.iterrows():
    if row["pssev"] in pssev_mapping:
        df_test.at[index, "pssev"] = pssev_mapping[row["pssev"]]
# print("pssev updated", df_test["pssev"])

print("pssev mapping applied to df_test")

# rename pimgres column

rename_mapping = {
    1: "specific changes",
    2: "unspecific changes",
    3: "no progression since last imaging",
    0: "normal",
}

# Applica la sostituzione dei valori nella colonna rinominata
df_test["pimgres"] = df_test["pimgres"].map(rename_mapping)
# print("pimgres updated", df_test["pimgres"])
print("pimgres mapping applied to df_test")

"""
rename all the columns of the df
"""
# rename the columns of the df_``
# apply the mapping to the columns to substitute the current name with the one in the file
mapping_path = os.path.join(
    saved_result_path, "mapping", "mapping_variables_names.xlsx"
)
mapping_df = pd.read_excel(mapping_path)

# look for the columns in the mapping file in the column variable and substitute it with the name in 'label' column
variable_to_label = dict(zip(mapping_df["variable"], mapping_df["label"]))
df_test.rename(columns=variable_to_label, inplace=True)

print("Columns renamed in df_test:\n", df_test.columns)

print(df_test)
# save in excel the df_test
df_test.to_csv(os.path.join(EXPERIMENT_PATH, f"df_test_best_fp_fn.csv"))

print("df_test saved in df_test_best_fp_fn.csv")

"""
FEATURE IMPORTANCE



# importances = results_dict["importances"]
feature_importance_data = results["feature_importance_data"]
feature_importances = feature_importance_data["feature_importances"]
top_10_features = feature_importance_data["top_10_features"]

# print("feature_importances:", feature_importances)
# print("top_10_features:", top_10_features)

# apply the mapping to the columns to substitute the current name with the one in the file
mapping_path = os.path.join(
    saved_result_path, "mapping", "mapping_variables_names.xlsx"
)
mapping_df = pd.read_excel(mapping_path)

# look for the columns in the mapping file in the column variable and substitute it with the name in 'label' column
# Create a dictionary to map 'variable' to 'label'
variable_to_label = dict(zip(mapping_df["variable"], mapping_df["label"]))

# Rename the keys in the 'feature_importances' dictionary
renamed_feature_importances = {}
for variable, importance in feature_importances.items():
    if variable in variable_to_label:
        new_key = variable_to_label[variable]
    else:
        print(f"No label found for variable: {variable}")
        new_key = variable  # Use the original name if no mapping is found
    renamed_feature_importances[new_key] = importance

# Sort the renamed_feature_importances by importance value in descending order
sorted_renamed_feature_importances = dict(
    sorted(
        renamed_feature_importances.items(), key=lambda item: item[1], reverse=True
    )
)

# Update feature_importance_data with sorted renamed dictionary
feature_importance_data["feature_importances"] = sorted_renamed_feature_importances

# Sort feature_importances and top_10_features by importance value in descending order
sorted_top_10_features = dict(
    sorted(top_10_features.items(), key=lambda item: item[1], reverse=True)
)

# Update feature_importance_data with sorted dictionaries
feature_importance_data.update(
    {
        "feature_importances": sorted_renamed_feature_importances,
        "top_10_features": sorted_top_10_features,
    }
)
"""
"""
PLOTS OF FP, FN
"""

"""
HISTOGRAMS OF CLINICAL DIAGNOSIS FROM TEST SET
"""

# Rename 'Isolated mitochondrial Cardiomyopathy' to 'Isolated Cardiomyopathy' in the whole DataFrame
df_test["Clinical Diagnosis"] = df_test["Clinical Diagnosis"].replace(
    "Isolated mitochondrial Cardiomyopathy", "Isolated Cardiomyopathy"
)

print("df_test imported")
print(df_test.columns)

# print(df_test['fp'])
print(df_test.head())

# print(df_test['Clinical Diagnosis'].value_counts())

# Select rows of Clinical Diagnosis where fp column is True
df_fp = df_test[df_test["fp"] == True]["Clinical Diagnosis"]
# remove the nan values
df_fp = df_fp.dropna()

# Select rows of Clinical Diagnosis where fn column is True
df_fn = df_test[df_test["fn"] == True]["Clinical Diagnosis"]
# remove the nan values
df_fn = df_fn.dropna()

# Count values for FP and FN
fp_counts = df_fp.value_counts()
fn_counts = df_fn.value_counts()

# put in other the values with less than    1 occurences
fp_counts["Other"] = fp_counts[fp_counts < 2].sum()
fn_counts["Other"] = fn_counts[fn_counts < 2].sum()

# Keep only counts greater than 1
fp_counts = fp_counts[fp_counts > 1]
fn_counts = fn_counts[fn_counts > 1]

# order in descending order
fp_counts = fp_counts.sort_values(ascending=False)
fn_counts = fn_counts.sort_values(ascending=False)

#
print("counter")
print(fp_counts)
print(fn_counts)

# Calculate percentages
fp_percent = (fp_counts / fp_counts.sum()) * 100
fn_percent = (fn_counts / fn_counts.sum()) * 100

# Order in descending order
fp_percent = fp_percent.sort_values(ascending=False)
fn_percent = fn_percent.sort_values(ascending=False)

print("percentage fp")
print(fp_percent)
print("percentage fn")
print(fn_percent)

# Plot for False Positives (FP)
plt.figure(figsize=(14, 7))

# Plot for False Positives (FP)
plt.subplot(1, 2, 1)
fp_percent.plot(kind="bar", color="#1f77b4")  # Dark blue
# plt.title('False Positives (FP)', fontsize=16)
plt.xlabel("Clinical Diagnosis", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Plot for False Negatives (FN)
plt.subplot(1, 2, 2)
fn_percent.plot(kind="bar", color="#aec7e8")  # Light blue
# plt.title('False Negatives (FN)', fontsize=16)
plt.xlabel("Clinical Diagnosis", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(
    os.path.join(EXPERIMENT_PATH, "fp_fn_hist.png"),
    format="png",
    bbox_inches="tight",
)
plt.close()





# print end of the script
print("End of the script")
