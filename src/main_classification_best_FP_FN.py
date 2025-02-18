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

# Constants and Paths
#GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_no_symp.csv")
BEST_PATH = os.path.join(saved_result_path_classification, "best_model")
EXPERIMENT_PATH = os.path.join(
    saved_result_path_classification, "experiments_all_models"
)
VERSION = "20250218_154057"  # best model version

#ask if consider patients with no sympthoms
Input=input("Do you want to consider patients with no symptoms? (y/n)")
if Input=="y":
    #GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed_all.csv")
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_symp.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_all"
    )
    BEST_PATH = os.path.join(saved_result_path_classification, "best_model_all")
    VERSION = "20250218_154604"


def setup_output(current_datetime):
    """Set up output redirection to a log file."""
    # file_name = f"classification_reports_{current_datetime}_no_mrmr.txt"
    # ask if clf or res
    file_name = f"classification_analysis_FP_FN.txt"
    sys.stdout = open(os.path.join(BEST_PATH, file_name), "w")


def get_gendna_type(value):
    if value in [4, 6, 8]:
        return "nDNA"
    elif value in [5, 7]:
        return "mtDNA"
    else:
        return None  # In case there are values outside the expected ones


def main():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_output(current_datetime)
    # Load and preprocess data
    df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH)

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
    ) = experiment_definition(X, y, df, EXPERIMENT_PATH, mt_DNA_patients)

    """
    MISCLASSIFICATION FN,FP ANALYSIS  
    """

    # Load the results and print them
    results_file_path = os.path.join(BEST_PATH, "best_model_results.pkl")
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)

    # identify the test subjects in the df raw
    df_raw = pd.read_csv(GLOBAL_DF_PATH)
    df_raw_test = df_raw[df_raw["subjid"].isin(test_subjects)]


    print("Results:")
    print(results)

    y_pred = results["y_pred"]
    y_test = results["y_test"]

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

    # print(df_raw_test)

    # print("False Positive (FP) Subject IDs:")
    # print(fp_subjids.values)

    # print("False Negative (FN) Subject IDs:")
    # print(fn_subjids.values)
    '''

    # add gendna column from df_not_numerical to the df_raw_test basing on subjid
    df_raw_test["gendna_non_num"] = df_raw_test["subjid"].map(
        df_raw.set_index("subjid")["gendna"]
    )
    # print the gendna of df_raw_test
    print("gendna of the df_raw_test:", df_raw_test["gendna_non_num"].values)

    # Apply the function to the 'gendna_non_num' column to create the 'gendna_type' column
    df_raw_test["gendna_type"] = df_raw_test["gendna_non_num"].apply(get_gendna_type)

    # print columns of the df_raw_test
    print("columns of the df_raw_test:", df_raw_test.columns)

    # print the df_raw_test the gendna_type for the fp_subjids and fn_subjids
    print("Class 0 mtDNA (5,7), Class 1 nDNA (4,6,8)")
    print("False Positive (FP) class and gendna_non_num, in reality mt 5,7:")
    print(df_raw_test.loc[fp_indices, "gendna_type"].values)
    # print(df_raw_test.loc[fp_indices, "gendna"].values)
    print(df_raw_test.loc[fp_indices, "gendna_non_num"].values)
    print("False Negative (FN) class and gendna_non_num, in reality n 4,6,8:")
    print(df_raw_test.loc[fn_indices, "gendna_type"].values)
    # print(df_raw_test.loc[fn_indices, "gendna"].values)
    print(df_raw_test.loc[fn_indices, "gendna_non_num"].values)

    print("df_raw_test shape:", df_raw_test.shape)
    print("df_raw_test columns:", df_raw_test.columns)
    print("df_raw_test head():", df_raw_test.head())
    '''

    # input("Press Enter to continue...")

    # save in excel the df_raw_test
    # df_raw_test.to_excel(os.path.join(saved_result_path_classification, f"df_raw_test_best.xlsx"))
    # print("df_raw_test numerical df saved in df_raw_test_best.xlsx")

    # create from the df_not_numerical the df with the subjid and the gendna
    # Drop columns from df_not_numerical
    # remove from columns_to_drop the columns_to_add

    """
    Creation of the df_test not numerical with also additional columns
    """

    # print shape of df_test
    print("Added to df_test the FP, FN and y_pred :", df_raw_test.shape)

    # print("df_test head():", df_test.head())

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
    df_test.to_csv(os.path.join(BEST_PATH, f"df_test_best_fp_fn.csv"))

    print("df_test saved in df_test_best_fp_fn.csv")

    """
    FEATURE IMPORTANCE

    """

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
        os.path.join(BEST_PATH, "fp_fn_hist.png"),
        format="png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

    '''
    # create a df of the best 27 features as columns and the subjid as index, selecting the columns from the saved_result_path_classification, "df_test.csv"
    # df_test_path = os.path.join(saved_result_path_classification, "df_test.csv")
    # df_test = pd.read_csv(df_test_path, index_col=0)
    # order in decreasing order the features
    df_test_important_features = df_test[
        list(sorted_renamed_feature_importances.keys())[
            :20
        ]  # TODO update threshold, to stop when the increment is less than 2%
    ]
    # add subjid to the df
    df_test_important_features["subjid"] = df_test.index
    # save the df
    df_test_important_features.to_csv(
        os.path.join(saved_result_path_classification, "df_test_important.csv")
    )

    # print("important columns of the df_test_importance:", df_test_important_features.columns)
    print("df_test_important features saved in df_test_important.csv")

    """
    Print table with the number of features and the percentage of explained importance
    """
    importances = sorted_renamed_feature_importances.values()

    # Initialize lists for the x and y coordinates of the plot
    num_features_used = []
    percentage_explained = []

    # Calculate the total importance
    total_importance = sum(importances)

    # Calculate the percentage of explained importance for each number of features
    cumulative_importance = 0
    for i, importance in enumerate(importances, 1):
        cumulative_importance += importance
        percentage = (cumulative_importance / total_importance) * 100
        num_features_used.append(i)
        percentage_explained.append(round(percentage, 2))  # Round to two decimal places
        # percentage_explained.append(percentage)

    # Prepare data for printing
    data = []
    for j, (feature_name, importance) in enumerate(
        sorted_renamed_feature_importances.items()
    ):
        data.append([j + 1, feature_name, percentage_explained[j]])

    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=["Number of Features", "Last feature added", "Percentage Explained"],
    )

    print("data of the progress of the importance:")
    # Print DataFrame
    print(df)

    # Save the df of the progress of the importance
    df.to_excel(
        os.path.join(
            saved_result_path_classification, f"feature_importance_progress.xlsx"
        )
    )

    # plot the percentage_explained in function of the num_features_used
    plt.figure(figsize=(10, 8))
    plt.plot(num_features_used, percentage_explained, marker="o")
    plt.title("Percentage of Explained Importance vs Number of Features", fontsize=15)
    plt.xlabel("Number of Features", fontsize=15)
    plt.ylabel("Percentage of Explained Importance", fontsize=15)
    # use a finer grid
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    feature_importance_file = f"Importance_distribution_percentage.png"
    plt.savefig(
        os.path.join(saved_result_path_classification, feature_importance_file),
        format="png",
        bbox_inches="tight",
    )
    plt.close()
    '''
