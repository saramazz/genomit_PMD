# Copyright 2023 Sara Mazzucato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time

### Third-party Library Imports:
import numpy as np
import pandas as pd

### Local Imports:
from config import global_path, saved_result_path
from utilities import *
from processing import *
from preprocessing import *
from plotting import *

"""
This script handles the creation, merging, and preprocessing of datasets from multiple hospitals.
Data is numerically encoded and saved in Excel for further analysis.
"""

print("Defining dataset path...")
dataset_path = os.path.join(global_path, "dataset")

print("Retrieving list of folders in dataset path...")
all_items = os.listdir(dataset_path)
folders = [
    item for item in all_items if os.path.isdir(os.path.join(dataset_path, item))
]
folders_to_remove = [
    "excel files",
    "Old versions",
    "Data Table Dictionaries",
    "Brescia",
]
hospital_names = [folder for folder in folders if folder not in folders_to_remove]
print("Hospital names:", hospital_names)

print("Reading important variables from Excel...")
important_vars_path = os.path.join(
    global_path, "variables_mapping/important_variables.xlsx"
)
df_vars = read_important_vars(important_vars_path)
important_vars_list, form_names = get_important_variables(important_vars_path)
common_vars_list = find_common_important_vars(
    hospital_names, form_names, global_path, df_vars
)

print("Processing data for each hospital...")
for hospital_name in hospital_names:
    dfs = []
    problem_forms = []

    print(f"\nStarting analysis for hospital: {hospital_name}")
    time.sleep(1)  # Reduced wait time for debugging

    for form_name in form_names:
        try:
            print(f"Processing form: {form_name} for hospital: {hospital_name}")

            form_path = os.path.join(
                global_path, "dataset", hospital_name, f"genomit-{form_name}.csv"
            )
            df_form = pd.read_csv(form_path, sep="\t")

            if "consent" in form_name:
                n_patients_consent = df_form["subjid"].nunique()

            vars = get_form_vars(form_name, df_vars, common_vars_list)
            if "consent" in form_name:
                df_form = df_form[vars]
            else:
                try:
                    df_form = df_form[vars + ["subjid", "visdat"]]
                except Exception as e:
                    problem_forms.append(form_name)
                    print(
                        f"Error processing file: {form_name} for {hospital_name}: {e}"
                    )
                    time.sleep(1)
                    continue

            print(f"Compiling data for each patient in form: {form_name}")
            unique_patients = df_form["subjid"].unique()
            for subjid in unique_patients:
                subjid_df = df_form[df_form["subjid"] == subjid]

                if subjid not in df["subjid"].values:
                    df = pd.concat([df, subjid_df])

                subjid_df, vars = process_form(subjid_df, form_name, vars)
                df.loc[df["subjid"] == subjid, vars] = subjid_df[vars].values

            print(f"All subjects processed for form: {form_name}")

        except FileNotFoundError:
            print(f"File for form {form_name} not found. Skipping...")

    print("Dropping duplicates from the DataFrame...")
    df.drop_duplicates(inplace=True)
    dfs.append(df)

    nRow, nCol = df.shape
    print(f"Processed DataFrame for {hospital_name}: {nRow} rows, {nCol} columns")
    print(f"Unique 'subjid' in final_df: {df['subjid'].nunique()}")
    print(f"Unique 'subjid' in consent: {n_patients_consent}")

    print(f"Saving DataFrame for {hospital_name} to disk...")
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_pickle(os.path.join(saved_result_path, "df", f"df_{hospital_name}.pkl"))

print("Merging DataFrames from all hospitals...")
merge_patient_dataframes(hospital_names, saved_result_path)

for hospital_name in hospital_names:
    pickle_file_path = os.path.join(saved_result_path, "df", f"df_{hospital_name}.pkl")
    df = pd.read_pickle(pickle_file_path)
    print(f"Total unique 'subjid' in {hospital_name}: {df['subjid'].nunique()}")
    nRow, nCol = df.shape
    print(f"DataFrame dimensions for {hospital_name}: {nRow} rows, {nCol} columns")

print("Loading and previewing global DataFrame...")
pickle_file_path = os.path.join(saved_result_path, "df", "df_Global.pkl")
df = pd.read_pickle(pickle_file_path)
nRow, nCol = df.shape
print(f"Global DataFrame dimensions: {nRow} rows, {nCol} columns")
print("Preview of df Global: ")
print(df.head())

print("Starting general preprocessing steps...")
nan_counts = count_nan_per_column(df)
print(f"NaN value count per column: {nan_counts}")

total_patients = calculate_total(
    global_path, form_names, hospital_names, calc_type="patients"
)
print(f"Total patients calculated: {total_patients}")

print("Removing prefixes from specific columns...")
df = remove_hp_prefix(df, "psterm__decod")
df = add_top5_symptoms_columns(df, hospital_names, "consent", "pmdsymptoms", "pmd")
df = remove_hp_prefix(df, "symp_on_")

nRow, nCol = df.shape
print(f"DataFrame dimensions after removing prefixes: {nRow} rows, {nCol} columns")

print("Applying specific preprocessing steps...")
columns_to_set_zero = [
    "abbi",
    "bgc",
    "ca",
    "cbea",
    "hisc",
    "inf",
    "le",
    "mrisll",
    "oa",
    "sh",
]
df.loc[df["pimgtype"] != 2, columns_to_set_zero] = 0

df = df[df["pmdno"] != 1]
if df["pmdno"].isnull().all():
    df = df.drop("pmdno", axis=1)
else:
    unique_values = df["pmdno"].unique()
    print(f"The 'pmdno' column contains the following unique values: {unique_values}")

df.loc[df["clindiag__decod"] == "C01", "sll"] = 1

print("Plotting phenotype distribution...")
plot_phenotype_distribution(df, "clindiag__decod", saved_result_path)
process_datetime_column(df, "psstdat__c")
process_datetime_column(df, "visdat")

df = df[~df["gendna"].isin([2, 3])]
df["gene"].replace("MTHFR", np.nan, inplace=True)
df["gene"].replace("twincle", "TWNK", inplace=True)

print("Updating encephalopathy column based on mappings...")
mapping_symptoms_path = os.path.join(
    saved_result_path, "mapping/mapping_sympthoms.xlsx"
)
HPO_terms_encephalopathy_path = os.path.join(
    saved_result_path, "mapping/HPO_terms_encephalopathy_PL.xlsx"
)
df = update_encephalopathy_column(
    df, mapping_symptoms_path, HPO_terms_encephalopathy_path
)

print("Adding abnormalities columns...")
mapping_abnormalities_path = os.path.join(
    saved_result_path, "mapping/mapping_abnormalities_HPO.xlsx"
)
df = add_abnormalities_cols(df, mapping_abnormalities_path, mapping_symptoms_path)

print("Ordering columns alphabetically...")
df = order_columns_alphabetically(df, ["Hospital", "subjid", "visdat"])

print("Checking for duplicate patients...")
df = check_multiple_patients(df, "subjid", "visdat", saved_result_path)

print("Updating 'subjid' by appending 'hospital'...")
df["subjid"] = df.apply(lambda row: f"{row['subjid']}_{row['Hospital']}", axis=1)
df = df.drop_duplicates(subset=["subjid", "visdat"], keep="last")

print("Saving preprocessed DataFrame...")
df.to_pickle(saved_result_path + "/df/df_preprocessed_Global.pkl")

print("Creating Excel version of the DataFrame...")
df.to_excel(saved_result_path + "/df/df_Global.xlsx", index=False)

print("Converting DataFrame to numerical format...")
df_num = convert_to_numerical(df)
df_num.to_pickle(os.path.join(saved_result_path, "df", "df_num_Global.pkl"))
df_num.to_excel(saved_result_path + "/df/df_Global_num.xlsx", index=False)

print("Plotting missing values...")
plot_missing_values(df, hospital_name, saved_result_path)
plot_missing_values_reduced(df, hospital_name)
plot_missing_values_diff(df, hospital_name, saved_result_path)

print("Script execution completed.")
