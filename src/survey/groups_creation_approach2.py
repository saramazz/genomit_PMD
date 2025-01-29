"""
Script to create groups for the survey and save in clinitian_patient_mapping.csv
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from collections import defaultdict
import time

# Import Streamlit libraries (if applicable)
import streamlit as st
from streamlit import session_state as ss

# put seed to have reproducible results
random.seed(42)

"""
DATA IMPORT AND INITIALIZATION
"""
# Paths setup
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(script_directory)
GLOBAL_PATH = os.path.dirname(parent_path)


saved_result_path = os.path.join(GLOBAL_PATH, "saved_results")
SURVEY_PATH = os.path.join(saved_result_path, "survey")

groups_path = os.path.join(SURVEY_PATH, "clinician_patient_mapping.csv")


df_path = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
df_test_path = os.path.join(SURVEY_PATH, "df_test_best.xlsx")
important_vars_path = os.path.join(
    GLOBAL_PATH, "variables_mapping", "important_variables.xlsx"
)

# Load main DataFrame
try:
    df_raw = pd.read_csv(df_path)
    print(f"Loaded DataFrame: {df_path}")
except FileNotFoundError:
    print(f"Error: File not found at {df_path}")
    exit()

# Display the dimensions and columns of the DataFrame
nRow, nCol = df_raw.shape
print(
    f"The DataFrame df_raw 'df_preprocessed_Global' contains {nRow} rows and {nCol} columns."
)

# Load test DataFrame
try:
    df_test = pd.read_excel(df_test_path)
    print(f"Loaded DataFrame: {df_test_path}")
except FileNotFoundError:
    print(f"Error: File not found at {df_test_path}")
    exit()

nRow, nCol = df_test.shape
print(f"The DataFrame 'df_test' contains {nRow} rows and {nCol} columns.")

"""
FILTER AND ANALYZE COLUMNS
"""
# Filter `df_raw` by `subject_id_test` and display new shape
subject_id_test = df_test["Subject id"].tolist()
df_raw = df_raw[df_raw["subjid"].isin(subject_id_test)]
print(f"Filtered df_raw dimensions: {df_raw.shape}")

# Round `cage` column to integers, handling NaN values
df_raw["cage"] = pd.to_numeric(df_raw["cage"], errors="coerce")  # Convert to numeric
df_raw["cage"] = (
    df_raw["cage"].apply(lambda x: round(x) if pd.notna(x) else x).astype("Int64")
)

# Calculate `cage` distribution
cage_distribution = df_raw["cage"].value_counts().to_dict()
print(f"Cage distribution: {cage_distribution}")

# Statistics for `cage`
print(
    f"Cage statistics: Mean = {df_raw['cage'].mean()}, Median = {df_raw['cage'].median()}, Std = {df_raw['cage'].std()}"
)

# Check for missing values in `cage` and `aao`
missing_cage = df_raw["cage"].isnull().sum()
missing_aao = df_raw["aao"].isnull().sum()
print(f"Missing values - Cage: {missing_cage}, AAO: {missing_aao}")

# Count rows where both `cage` and `aao` are missing
missing_both = df_raw[df_raw["cage"].isnull() & df_raw["aao"].isnull()].shape[0]
print(f"Rows missing both `cage` and `aao`: {missing_both}")

# Display `aao` and `cage` columns
print("Preview of 'aao' and 'cage' columns:")
print(df_raw[["aao", "cage"]])

"""
CLASSIFY PATIENTS BY AGE GROUP
"""
# Classify patients as 'adult' or 'young' based on `cage` and `aao`
df_raw["patient_class"] = "adult"
df_raw.loc[(df_raw["cage"].isnull()) & (df_raw["aao"] <= 16), "patient_class"] = "young"
df_raw.loc[df_raw["cage"] <= 16, "patient_class"] = "young"

# Display classification results
print(f"Patient classification distribution:\n{df_raw['patient_class'].value_counts()}")

# Check for missing values in `patient_class`
missing_class = df_raw["patient_class"].isnull().sum()
print(f"Missing values in 'patient_class': {missing_class}")

# Calculate percentages
adults_count = df_raw[df_raw["patient_class"] == "adult"].shape[0]
young_count = df_raw[df_raw["patient_class"] == "young"].shape[0]
total_count = df_raw.shape[0]

percentage_adults = (adults_count / total_count) * 100
percentage_young = (young_count / total_count) * 100

print(f"Percentage - Adults: {percentage_adults:.2f}%, Young: {percentage_young:.2f}%")

"""
CREATE TEST DATAFRAME WITH SELECTED FEATURES
"""

# Load important variables for classification
try:
    df_vars = pd.read_excel(important_vars_path)
    print(f"Loaded variables from: {important_vars_path}")
except FileNotFoundError:
    print(f"Error: File not found at {important_vars_path}")
    exit()

# Determine columns to drop
columns_to_drop = list(
    df_vars.loc[
        df_vars["consider for mtDNA vs nDNA classification?"] == "N", "variable"
    ]
)

# Additional columns to drop
additional_columns_to_drop = [
    "Hospital",
    "epiphen",
    "sll",
    "clindiag__decod",
    "encephalopathy",
    "gendna",
]
additional_columns_to_drop += [
    col for col in df_raw.columns if "pimgtype" in col or "psterm" in col
]

# Drop specified columns
columns_to_drop += additional_columns_to_drop
df = df_raw.drop(columns=columns_to_drop, errors="ignore").fillna(
    998
)  # fill missing values with 998

# Display the cleaned DataFrame's dimensions and columns
print(f"Cleaned DataFrame dimensions: {df.shape}")
print("Columns after cleaning:", df.columns)


# CREATE GROUPS FOR SURVEY

# create a list of id for the clinicians, from 0 to 39: 0,19 experts, 20,39 non-experts
clinicians_expert = range(20)
clinicians_non_expert = range(20, 40)

# associate to each patient at least for clinicians, two expert and two non-expert


# Function to randomly select n clinicians from a list
def select_random_clinicians(clinicians, n=2):
    return random.sample(clinicians, n)


# Assign two expert clinicians and two non-expert clinicians to each patient
df["clinician_expert"] = df["subjid"].apply(
    lambda _: select_random_clinicians(clinicians_expert, 2)
)
df["clinician_non_expert"] = df["subjid"].apply(
    lambda _: select_random_clinicians(clinicians_non_expert, 2)
)

# Verify that each patient is assigned exactly 2 expert and 2 non-expert clinicians
for _, row in df.iterrows():
    expert_count = len(row["clinician_expert"])
    non_expert_count = len(row["clinician_non_expert"])
    if expert_count != 2 or non_expert_count != 2:
        print(
            f"Error: Patient {row['subjid']} is not assigned to 2 expert and 2 non-expert clinicians."
        )
        break
else:
    print("All patients are assigned to 2 expert and 2 non-expert clinicians.")

# Display the updated DataFrame
print("Updated DataFrame with assigned clinicians:")
print(df.head())


# Create a mapping of clinician IDs to patient IDs
clinician_to_patients = defaultdict(list)

# Populate the mapping
for _, row in df.iterrows():
    for clinician in row["clinician_expert"]:
        clinician_to_patients[clinician].append(row["subjid"])
    for clinician in row["clinician_non_expert"]:
        clinician_to_patients[clinician].append(row["subjid"])

# print the clinician_to_patients
for clinician, patients in clinician_to_patients.items():
    print(f"Clinician {clinician}: {patients}")

# order by clinician id
clinician_to_patients = dict(sorted(clinician_to_patients.items()))


# Function to balance patients for each clinician
def balance_patients(clinician_to_patients, df, target_total=40):
    for clinician, assigned_patients in clinician_to_patients.items():

        # print the clinician id and the number of patients assigned to him
        print(f"Clinician {clinician}: Total Patients={len(assigned_patients)}")

        # Filter assigned patients
        assigned_df = df[df["subjid"].isin(assigned_patients)]

        # Count young and adult patients
        count_young = len(assigned_df[assigned_df["patient_class"] == "young"])
        count_adult = len(assigned_df[assigned_df["patient_class"] == "adult"])

        # Calculate the number of missing patients to reach the targets
        total_assigned = len(assigned_patients)
        missing_total = target_total - total_assigned

        target_young = int((target_total * percentage_young) / 100)
        target_adult = int((target_total * percentage_adults) / 100)
        missing_young = target_young - count_young
        missing_adult = (
            target_total - missing_young - total_assigned
        )  # to be sure they are 40 patients
        # total patients expected to be assigned to the clinician
        print(
            f"Total Patients expected to be assigned to the clinician={missing_adult+missing_young+total_assigned}"
        )
        # if it is different from 40, break and rise error
        if missing_adult + missing_young + total_assigned != 40:
            print(
                f"Error: The total number of patients assigned to clinician {clinician} is different from 40."
            )
            break

        # print the number of young and adult patients assigned to the clinician
        print(f"Assigned Young={count_young}, Assigned Adults={count_adult}")
        # print the number of young and adult patients that should be assigned to the clinician
        print(f"Target Young={target_young}, Target Adults={target_adult}")
        # print the missing young and adult patients
        print(f"Missing Young={missing_young}, Missing Adults={missing_adult}")

        # Add missing patients
        if missing_total > 0:
            # Available patients not yet assigned
            unassigned_df = df[~df["subjid"].isin(assigned_patients)]

            # Add young patients
            if missing_young > 0:
                young_candidates = unassigned_df[
                    unassigned_df["patient_class"] == "young"
                ]
                young_to_add = young_candidates.sample(
                    min(missing_young, len(young_candidates))
                )
                assigned_patients.extend(young_to_add["subjid"].tolist())

            # Add adult patients
            if missing_adult > 0:
                adult_candidates = unassigned_df[
                    unassigned_df["patient_class"] == "adult"
                ]
                adult_to_add = adult_candidates.sample(
                    min(missing_adult, len(adult_candidates))
                )
                assigned_patients.extend(adult_to_add["subjid"].tolist())

            # Update the mapping
            clinician_to_patients[clinician] = assigned_patients[
                :target_total
            ]  # Limit to 40 patients

        # check the final number of patients assigned to the clinician and the percentage of young and adult patients
        final_df = df[df["subjid"].isin(assigned_patients)]
        num_young = len(final_df[final_df["patient_class"] == "young"])
        num_adult = len(final_df[final_df["patient_class"] == "adult"])
        print(
            f"Final Total Patients={len(assigned_patients)}, Young={num_young}, Adults={num_adult}"
        )
        # rise an error if the number of patients assigned to the clinician is different from 40 or if the percentage of young and adult patients is different from the expected one
        if len(assigned_patients) != 40 or num_young > 11 or num_adult > 31:
            print(
                f"Error: The total number of patients assigned to clinician {clinician} is different from 40 or the percentage of young and adult patients is different from the expected one."
            )

            # stop the code
            exit()

    return clinician_to_patients


# check if groups_path = os.path.join(SURVEY_PATH, "clinician_patient_mapping.csv") exixt, orherwise proceed:

if os.path.exists(groups_path):
    print(f"File already exists: {groups_path}")
    exit()
else:
    print(f"File does not exist: {groups_path}")
    print("Proceeding with the creation of the groups...")
    # Balance patients for each clinician
    balanced_clinician_to_patients = balance_patients(clinician_to_patients, df)

    # Final counts and output
    for clinician, patients in balanced_clinician_to_patients.items():
        final_df = df[df["subjid"].isin(patients)]
        num_young = len(final_df[final_df["patient_class"] == "young"])
        num_adult = len(final_df[final_df["patient_class"] == "adult"])
        print(
            f"Clinician {clinician}: Total Patients={len(patients)}, Young={num_young}, Adults={num_adult}"
        )

    # print the final clinician_to_patients
    for clinician, patients in balanced_clinician_to_patients.items():
        print(f"Clinician {clinician}: {patients}")

    # save the result to a csv file, foe each clinician, the list of patients assigned to him
    output = []
    for clinician, patients in balanced_clinician_to_patients.items():
        for subjid in patients:
            output.append({"clinician_id": clinician, "subjid": subjid})
    # order by clinician id
    output = sorted(output, key=lambda x: x["clinician_id"])
    # save the result to a csv file
    output_df = pd.DataFrame(output)
    # date and time
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    # add the date and time to the file name
    file_name = f"clinician_patient_mapping.csv"
    output_df.to_csv(os.path.join(SURVEY_PATH, file_name), index=False)
    print(f"Saved clinician-patient mapping to: {os.path.join(SURVEY_PATH, file_name)}")


"""
CHECK THE FINAL RESULT
"""
print("Checking the final result...")
df = pd.read_csv(groups_path)

# print the dimension and the columns names of the data
print(df.shape)
print(df.columns)

# check how many subjid each clinician has associated and if it is different from 40 print it
print("Number of subjid associated to each clinician:")
print(df.groupby("clinician_id").size())
print(df.groupby("clinician_id").size().unique())
print(df.groupby("clinician_id").size().value_counts())

# check if there are duplicated subjid associated to the same clinician
print("Duplicated rows:")
print(df.duplicated(subset=["clinician_id", "subjid"]).sum())

# check the distribution of young and adult patients for each clinician
df = df.merge(df_raw[["subjid", "patient_class"]], how="left", on="subjid")
df["patient_class"] = df["patient_class"].astype("category")
df["patient_class"] = df["patient_class"].cat.reorder_categories(["young", "adult"])
df["patient_class"] = df["patient_class"].cat.as_ordered()
# print the distribution of young and adult patients for each clinician
print("Distribution of young and adult patients for each clinician:")
# unstack the result to have a better visualization
print(df.groupby(["clinician_id", "patient_class"]).size().unstack().fillna(0))
