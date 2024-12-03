"""
code to do classification of nDNA vs mtDNA experimenting features, balancement and classifiers
"""

import os
import sys
import time
from collections import Counter
from itertools import combinations
import json


### Third-party library imports:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut, KFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, mean_squared_error, f1_score, make_scorer
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import mode
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier


### Locally defined modules:

from config import global_path, saved_result_path_classification
from utilities import *
from processing import *
from plotting import *

"""
IMPORT DATA
"""

# Load Global DataFrame with only first visit data
hospital_name = "Global"
df_path = os.path.join(saved_result_path, "df", "df_preprocessed_Global.pkl")

df = pd.read_pickle(df_path)


# TODO: reduce the df
# df = df.head(50)

# Display the dimensions and columns of the DataFrame
nRow, nCol = df.shape
print(
    f'The DataFrame "df_preprocessed" from {hospital_name} hospital contains {nRow} rows and {nCol} columns.'
)
# print('Columns:', df.columns)

# print distribution of classes, drop nans and 1, convert to numerical and print class distribution
df, df_not_numerical = process_gendna_column(df)
time.sleep(70)

# insert Not applicable as -1
df = fill_missing_values(df)

print(df)

"""
FEATURE SELECTION
"""

# Load the important variables from Excel
important_vars_path = os.path.join(global_path, "dataset", "important_variables.xlsx")
df_vars = pd.read_excel(important_vars_path)

# Specify the column name for considering variables
column_name = "consider for mtDNA vs nDNA classification?"

# Get the list of columns to drop based on 'N' in the specified column
columns_to_drop = list(df_vars.loc[df_vars[column_name] == "N", "variable"])
print("Columns to drop:", columns_to_drop)

# Additional columns to drop
additional_columns_to_drop = [
    "Hospital",
    "nDNA",
    "mtDNA",
    "gendna_type",
    "epiphen",
    "sll",
    "clindiag__decod",
    "encephalopathy",
]
additional_columns_to_drop += [col for col in df.columns if "pimgtype" in col]
additional_columns_to_drop += [col for col in df.columns if "psterm" in col]

columns_to_drop = columns_to_drop + additional_columns_to_drop

print("Columns to drop:", columns_to_drop)


# Sostituisci i valori mancanti con 998
df = df.fillna(998)
df_raw = df.copy()  # save the raw data non numerical

# Drop the columns from the DataFrame and convert to numerical
X, y, X_df = define_X_y(df, columns_to_drop)


# Define experiment parameters and split the data saving in
(
    X_train,
    X_test,
    y_train,
    y_test,
    train_subjects,
    test_subjects,
    kf,
    scorer,
    thr,
    nFeatures,
    num_folds,
) = experiment_definition(X, y, X_df)

time.pause(70)

# remove subjid from X_df
X_df = X_df.drop(columns=["subjid"])
features = X_df.columns

# convert X to df
# features=pd.DataFrame(X).columns

print("Features names:", features)
print("X_df shape:", X_df.shape)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

time.sleep(10)
# paremeter for feature selection


# Redirect the standard output to a file
# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the file name with the current date and time
# file_name = f"classification_reports_{current_datetime}_SVM.txt"
file_name = f"classification_reports_{current_datetime}_ALL.txt"

# Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result_path_classification, file_name), "w")


# sys.stdout = open(os.path.join(saved_result_path_classification, "classification_reports_2703.txt"), 'w')
print("Starting the classification...")

"""
classifiers
"""
# Define the classifiers and their parameter grids

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
}

# define the settings of the experiment
balancing_techniques = [
    "no",
    "over",
    "under",
]  
feature_selection_options = ["no", "pca", "mrmr", "select_from_model", "rfe"]


for classifier, (clf_model, param_grid) in classifiers.items():
    print("********************************************************************\n")
    pipeline = Pipeline([("clf", clf_model)])  # create pipeline to clf

    for feature_selection_option in feature_selection_options:
        # Apply feature selection option to the data if necessary
        # Process feature selection
        X_train_selected, X_test_selected, param_grid_selected, pipeline_selected = (
            process_feature_selection(
                clf_model,
                X_df,
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
                X_df,
                X_train_bal,
                X_test_bal,
                y_train_bal,
                y_test_bal,
                kf,
                scorer,
                features,
                balancing_technique,
                feature_selection_option,
            )


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__

"""

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
