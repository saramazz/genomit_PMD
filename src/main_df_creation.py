import os
import time


### Third-party Library Imports:

import numpy as np
import pandas as pd


### Local Imports:

from config import global_path, saved_result_path
from utilities import *
from processing import *
from plotting import *

"""
This script handles the creation, merging, and preprocessing of datasets from multiple hospitals.
Data is numerically encoded and saved in Excel for further analysis.
"""

# Define dataset path
dataset_path = os.path.join(global_path, "dataset")

# Get list of all directories except specified ones
all_items = os.listdir(dataset_path)
folders = [item for item in all_items if os.path.isdir(os.path.join(dataset_path, item))]
folders_to_remove = ["excel files", "Old versions", "Data Table Dictionaries", "Brescia"]
hospital_names = [folder for folder in folders if folder not in folders_to_remove]
print("Hospital names:", hospital_names)

# Read important variables
important_vars_path = os.path.join(global_path, "dataset", "important_variables.xlsx")
df_vars = read_important_vars(important_vars_path)
important_vars_list, form_names = get_important_variables(important_vars_path)
common_vars_list = find_common_important_vars(hospital_names, form_names, global_path, df_vars)

# Processing each hospital's datasets
for hospital_name in hospital_names:
    # Initialize storage for DataFrames
    dfs = []
    problem_forms = []

    print(f'Starting analysis for hospital: {hospital_name}')
    time.sleep(5)  # Temporary wait time for debugging

    # Iterate over each form for the hospital
    for form_name in form_names:
        try:
            print(f'Processing form: {form_name} for hospital: {hospital_name}')
            
            # Build the form path
            form_path = os.path.join(global_path, 'dataset', hospital_name, f'genomit-{form_name}.csv')

            # Load the form data
            df_form = pd.read_csv(form_path, sep='\t') 

            # If form is consent type, note the number of unique patients
            if 'consent' in form_name:
                n_patients_consent = df_form['subjid'].nunique()

            vars = get_form_vars(form_name, df_vars, common_vars_list)
            if 'consent' in form_name:
                df_form = df_form[vars]
            else:
                try:
                    df_form = df_form[vars + ['subjid', 'visdat']]
                except:
                    problem_forms.append(form_name)
                    print(f'Error processing file: {form_name} for {hospital_name}')
                    time.sleep(5)
                    continue

            # Compile data for each patient
            unique_patients = df_form['subjid'].unique()
            for subjid in unique_patients:
                subjid_df = df_form[df_form['subjid'] == subjid]

                if subjid not in df['subjid'].values:
                    df = pd.concat([df, subjid_df])

                # Form-specific processing
                subjid_df, vars = process_form(subjid_df, form_name, vars)
                df.loc[df['subjid'] == subjid, vars] = subjid_df[vars].values

            print('All subjects processed for this form')
        
        except FileNotFoundError:
            print(f"File for form {form_name} not found. Skipping...")

    # Remove duplicates and save DataFrame
    df.drop_duplicates(inplace=True)
    dfs.append(df)

    # Compare dimensions of df and unique subjects in the consent form
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns in df for hospital: {hospital_name}')
    print(f"Total unique 'subjid' in final_df: {df['subjid'].nunique()}")
    print(f"Total unique 'subjid' in consent: {n_patients_consent}")

    # Save the final DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_pickle(os.path.join(saved_result_path, 'df', f'df_{hospital_name}.pkl'))

# Merge dataframes from all hospitals
merge_patient_dataframes(hospital_names, saved_result_path)

# Print the dimension of each hospital's DataFrame
for hospital_name in hospital_names:
    pickle_file_path = os.path.join(saved_result_path, "df", f"df_{hospital_name}.pkl")
    df = pd.read_pickle(pickle_file_path)
    print(f"Total unique 'subjid' in {hospital_name}: {df['subjid'].nunique()}")
    nRow, nCol = df.shape
    print(f"There are {nRow} rows and {nCol} columns in the df of {hospital_name}")

# Load and preview global dataframe
pickle_file_path = os.path.join(saved_result_path, "df", "df_Global.pkl")
df = pd.read_pickle(pickle_file_path)
nRow, nCol = df.shape
print(f"There are {nRow} rows and {nCol} columns in df_Global")
print("Preview of df Global: ")
print(df.head())

# General preprocessing steps
nan_counts = count_nan_per_column(df)

total_patients = calculate_total(global_path, form_names, hospital_names, calc_type="patients")

df = remove_hp_prefix(df, "psterm__decod")

df = add_top5_symptoms_columns(df, hospital_names, "consent", "pmdsymptoms", "pmd")

df = remove_hp_prefix(df, "symp_on_")

# Assuming the DataFrame is named df, and the "sex" column exists in it

nRow, nCol = df.shape
print(f"There are {nRow} rows and {nCol} columns in df_Global after hp prefix")

# Specific preprocessing
columns_to_set_zero = ["abbi", "bgc", "ca", "cbea", "hisc", "inf", "le", "mrisll", "oa", "sh"]
df.loc[df["pimgtype"] != 2, columns_to_set_zero] = 0

df = df[df["pmdno"] != 1]

if df["pmdno"].isnull().all():
    df = df.drop("pmdno", axis=1)
else:
    unique_values = df["pmdno"].unique()
    print(f"The 'pmdno' column contains the following unique values: {unique_values}")

df.loc[df["clindiag__decod"] == "C01", "sll"] = 1

plot_phenotype_distribution(df, "clindiag__decod", saved_result_path)
process_datetime_column(df, "psstdat__c")
process_datetime_column(df, "visdat")

df = df[~df["gendna"].isin([2, 3])]

df["gene"].replace("MTHFR", np.nan, inplace=True)
df["gene"].replace("twincle", "TWNK", inplace=True)

mapping_symptoms_path = os.path.join(saved_result_path, "mapping/mapping_sympthoms.xlsx")
HPO_terms_encephalopathy_path = os.path.join(saved_result_path, "mapping/HPO_terms_encephalopathy_PL.xlsx")
df = update_encephalopathy_column(df, mapping_symptoms_path, HPO_terms_encephalopathy_path)

mapping_abnormalities_path = os.path.join(saved_result_path, "mapping/mapping_abnormalities_HPO.xlsx")
df = add_abnormalities_cols(df, mapping_abnormalities_path, mapping_symptoms_path)

df = order_columns_alphabetically(df, ["Hospital", "subjid", "visdat"])

# Check for duplicates based on subjid and visdat
df = check_multiple_patients(df, "subjid", "visdat", saved_result_path)

# Update 'subjid' by appending 'hospital'
df["subjid"] = df.apply(lambda row: f"{row['subjid']}_{row['Hospital']}", axis=1)

df = df.drop_duplicates(subset=["subjid", "visdat"], keep="last")

# Save preprocessed DataFrame
df.to_pickle(saved_result_path + "/df/df_preprocessed_Global.pkl")

# Create Excel version of the DataFrame
df.to_excel(saved_result_path + "/df/df_Global.xlsx", index=False)

# Convert to numerical
df_num = convert_to_numerical(df)
df_num.to_pickle(os.path.join(saved_result_path, "df", "df_num_Global.pkl"))

# Create Excel version of the numerical DataFrame
df_num.to_excel(saved_result_path + "/df/df_Global_num.xlsx", index=False)

# Plot missing values
plot_missing_values(df, hospital_name, saved_result_path)
plot_missing_values_reduced(df, hospital_name)
plot_missing_values_diff(df, hospital_name, saved_result_path)