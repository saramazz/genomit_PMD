"""
Script to perform classification of nDNA vs mtDNA.
Evaluates the BEST classifier
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
BEST_PATH = os.path.join(saved_result_path_classification, "best_model")
EXPERIMENT_PATH = os.path.join(
    saved_result_path_classification, "experiments_all_models"
)
VERSION = "20250128_165035"  # best model version

# Ensure necessary directories exist
os.makedirs(BEST_PATH, exist_ok=True)


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
    print("Saving path:", BEST_PATH)

    print("Dataset for classification shape:", df.shape)

    print("Dimension of X_train:", X_train.shape)
    print("Dimension of X_test:", X_test.shape)
    print("Dimension of y_train:", y_train.shape)
    print("Dimension of y_test:", y_test.shape)
    print("Features names:", features)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""
    # file_name = f"classification_reports_{current_datetime}_no_mrmr.txt"
    file_name = f"classification_reports_best.txt"
    sys.stdout = open(os.path.join(BEST_PATH, file_name), "w")


def perform_classification_best(X_test, y_test, best_estimator, results_path):
    # Evaluate on the test set
    y_pred = best_estimator.predict(X_test)

    # Calculate performance metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Store results
    results_to_save = {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "classification_report": class_report,
    }

    # Create results directory if needed and save pickled results
    os.makedirs(results_path, exist_ok=True)
    results_file_path = os.path.join(results_path, "best_model_results.pkl")
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)

    print(f"Results saved to {results_file_path}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")

    return results_to_save


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
        test_subjects,
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

    """
    df test creation
    """

    """
    Creation of the df_test not numerical with also additional columns
    """
    columns_to_add = [
        "clindiag__decod",
        "gendna",
        "gene",
        "nminh",
        "cmut",
        "mtpos",
        "subjid",
    ]

    columns_to_drop = [col for col in columns_to_drop if col not in columns_to_add]

    df_test = df[df["subjid"].isin(test_subjects)]  # create df with not numerical data

    # Drop columns from df_test
    df_test.drop(columns=columns_to_drop, inplace=True)

    print(
        "df_not_numerical shape after gendna processing, filling NAs, dropping columns of X and adding some for more explainability:",
        df.shape,
    )

    # print("df_test head():", df_test.head())

    """
    sobstitute HPO code with the name of the symptom
    """

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
    df_test.to_excel(os.path.join(BEST_PATH, f"df_test_best.xlsx"))

    print("df_test saved in df_test_best.xlsx")

    print("Starting the classification...")

    best_model_path = os.path.join(EXPERIMENT_PATH, f"best_classifier_{VERSION}.pkl")
    with open(best_model_path, "rb") as f:
        best_classifiers = pickle.load(f)

    # print the type and size of the best_classifiers
    print("Type of best_classifiers:", type(best_classifiers))
    print("Size of best_classifiers:", len(best_classifiers))

    # If the tuple has a consistent format as expected
    if isinstance(best_classifiers, tuple):
        # Unpack and examine item types
        classifier_name, best_estimator, best_score, results = best_classifiers
        print(f"Classifier Name: {classifier_name}")
        print(f"Best Estimator: {best_estimator}")
        print(f"Best Score: {best_score}")
        # print(f"Additional Results: {results}")

        results = best_classifiers[3]
        best_params = results["best_params"]

        print("Best Estimator:")
        print(best_estimator)
        print("Best Parameters:")
        print(best_params)

    # Create an imputer object with 'constant' strategy and your specified fill value
    imputer = SimpleImputer(strategy="constant", fill_value=998)

    # Fit the imputer on the training data and transform it
    X_train = imputer.fit_transform(X_train)
    # Transform the test data with the same imputer
    X_test = imputer.transform(X_test)

    print("X_test")
    print(X_test)

    perform_classification_best(X_test, y_test, best_estimator, BEST_PATH)

    print("Classification with the best classifier completed and results saved.")


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
