

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

# Constants and Paths
GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
SURVEY_PATH = os.path.join(
    saved_result_path, "survey"
)
CLF_RESULTS_PATH = os.path.join(saved_result_path_classification,"saved_data")

# Ensure necessary directories exist
os.makedirs(SURVEY_PATH, exist_ok=True)

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

def setup_output(current_datetime):
    """Set up output redirection to a log file."""

    # file_name = f"classification_reports_{current_datetime}_mrmr.txt"
    file_name = f"survey_log.txt"  # {current_datetime}_mrmr.txt"
    sys.stdout = open(os.path.join(SURVEY_PATH, file_name), "w")

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
        train_subjects,
        test_subjects,
        features,
        kf,
        scorer,
        thr,
        nFeatures,
        num_folds,
    ) = experiment_definition(X, y, df, SURVEY_PATH)

    #save X, y and the df in /home/saram/PhD/genomit_PMD/saved_results/classifiers_results/saved_data as pkl
    joblib.dump(X, os.path.join(CLF_RESULTS_PATH, "X.pkl"))
    joblib.dump(y, os.path.join(CLF_RESULTS_PATH, "y.pkl"))

    #print dimension of X and y
    print("Dimension of X:", X.shape)
    print("Dimension of y:", y.shape)

    #input("Press Enter to start the df_test creation...")


    """
        Creation of the df_test not numerical with also additional columns
    """

    #test_subjects = df.loc[df["subjid"].isin(X_test.index), "subjid"]

    #print dimension of test subjects
    print("Dimension of test subjects:", test_subjects.shape)

    #print the test subjects
    #print("Test subjects:", test_subjects)



    #add the cols to the df_test from the original df
    df_raw= pd.read_csv(GLOBAL_DF_PATH)
    
    #print dimension of df_raw
    print("Dimension of df_raw:", df_raw.shape)

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
        col for col in df_raw.columns if "pimgtype" in col or "psterm" in col
    ]
    columns_to_drop += additional_columns  


    df_test = df_raw[df_raw["subjid"].isin(test_subjects)]  # create df with not numerical data

    #print dimension of column to drop
    print("Dimension of columns to drop:", len(columns_to_drop))
    #only consider the common columns between df_raw and columns_to_drop
    columns_to_drop = [col for col in columns_to_drop if col in df_test.columns]
    #print the columns to drop
    print("Common Columns to drop:", columns_to_drop)

    #print the non common columns between df_raw and columns_to_drop
    print("Non common Columns to drop:", [col for col in columns_to_drop if col not in df_test.columns])

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
        "E": "Unspecified mitochondrial disorder",
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

    #df_test["clindiag__decod"] = df_test["clindiag__decod"].map(clindiag_mapping)

    # print("clin diag decod updated", df_test["clindiag__decod"])

    # print("Columns added to df_test:\n", df_test[columns_to_add])
    # print("size of df_test", df_test.shape)
    #print("clindiag__decod mapping applied to df_test")

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
    df_test.to_excel(os.path.join(SURVEY_PATH, f"df_test_best.xlsx"))

    print("df_test saved in df_test_best.xlsx")


if __name__ == "__main__":
    main()