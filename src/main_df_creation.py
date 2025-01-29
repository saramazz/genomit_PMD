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
from config import global_path, saved_result_path, important_vars_path
from utilities import *
from processing import *
from preprocessing import *
from plotting import *

"""
This script handles the creation, merging, and preprocessing of datas from multiple hospitals.
Data is numerically encoded and saved in Excel for further analysis.
"""


print("Defining data path...")
data_path = os.path.join(global_path, "data")
saved_result_path = os.path.join(saved_result_path, "df")


print("Retrieving list of folders in data path...")
all_items = os.listdir(data_path)
folders = [item for item in all_items if os.path.isdir(os.path.join(data_path, item))]

# print the folders
# print("Folders in data path:", folders)
folders_to_remove = ["Genova", "Brescia"]  ##TODO include these hospitals
hospital_names = [folder for folder in folders if folder not in folders_to_remove]
# print the number of hospitals
print(f"Number of hospitals: {len(hospital_names)}")
print("Hospital names:", hospital_names)

input("Press Enter to proceed with the script...")


print("Creating DataFrame for all hospitals...")
print("Reading important variables from Excel...")

df_vars = read_important_vars(important_vars_path)
important_vars_list, form_names = get_important_variables(important_vars_path)
common_vars_list = find_common_important_vars(
    hospital_names, form_names, global_path, df_vars
)

# stop asking an input if proceed
# input("Press Enter to proceed with the script...")

print("Processing data for each hospital...")

dfs = []
problem_forms = []

max_symps = 16

for hospital_name in hospital_names:
    print(f"Starting analysis for hospital {hospital_name}")

    # Path where the DataFrame for this hospital would be saved
    hospital_df_path = os.path.join(saved_result_path, f"df_{hospital_name}.pkl")

    # Check if data for this hospital already exists
    if os.path.exists(hospital_df_path):
        print(f"Data for {hospital_name} already processed. Skipping...")
        # Optionally load this hospital's DataFrame to include in final_df
        existing_df = pd.read_pickle(hospital_df_path)
        dfs.append(existing_df)
        continue

    df = pd.DataFrame(columns=common_vars_list + ["epiphen", "sll"])

    for i in range(max_symps):
        df[f"psterm__decod_{i}"] = np.nan

    form_path = os.path.join(data_path, "Besta", f"genomit-pmdimagingresultlog.csv")
    df_pimagingresultlog = pd.read_csv(form_path, sep="\t", encoding="ISO-8859-1")

    unique_pimgtypes = sorted(df_pimagingresultlog["pimgtype"].dropna().unique())
    for pimgtype in unique_pimgtypes:
        df[f"pimgtype_{pimgtype}"] = np.nan

    data_list = []

    for form_name in form_names:
        try:
            print(f"Processing form_name: {form_name} of {hospital_name}")

            common_path = os.path.join(data_path, hospital_name)
            form_path = os.path.join(common_path, f"genomit-{form_name}.csv")
            df_form = pd.read_csv(form_path, sep="\t")

            if "consent" in form_name:
                n_patients_consent = df_form["subjid"].nunique()

            vars = get_form_vars(form_name, df_vars, common_vars_list)

            if "consent" in form_name:
                df_form = df_form[vars]
            else:
                try:
                    df_form = df_form[vars + ["subjid", "visdat"]]
                except:
                    problem_forms.append(form_name)
                    print(f"Please check the file: {form_name} of {hospital_name}")
                    time.sleep(5)
                    continue

            unique_patients = df_form["subjid"].unique()
            repeated_patients = (
                df_form["subjid"].value_counts().loc[lambda x: x > 1].index.tolist()
            )

            for subjid in df_form["subjid"]:
                vars = get_form_vars(form_name, df_vars, common_vars_list)
                subjid_df = df_form[df_form["subjid"] == subjid]

                if subjid not in df["subjid"].values:
                    df = pd.concat([df, subjid_df[subjid_df["subjid"] == subjid]])

                    if form_name != "consent":
                        entry = {
                            "subjid": subjid,
                            "form": form_name,
                            "hospital_name": hospital_name,
                        }
                        data_list.append(entry)

                subjid_df, vars = process_form(subjid_df, form_name, vars, max_symps)

                for column in vars:
                    column_values = subjid_df[column].values
                    last_value = "NA" if len(column_values) == 0 else column_values[-1]
                    df.loc[df["subjid"] == subjid, column] = last_value

            print("All subjects were added in 1 line for this form.")

        except FileNotFoundError:
            print(f"File for form_name {form_name} not found. Skipping...")
    # Add a hospital column to track the origin of each row
    df["Hospital"] = hospital_name  # Add the hospital column here

    df.drop_duplicates(inplace=True)
    dfs.append(df)

    subjid_form_df = pd.DataFrame(data_list)
    #save the subjid_form_df in the saved_result_path as xlsx
    subjid_form_df.to_excel(os.path.join(saved_result_path, f"subjid_form_info_{hospital_name}.xlsx"), index=False)


    

    nRow, nCol = df.shape
    print(f"There are {nRow} rows and {nCol} columns in the df_{hospital_name}")
    print(f"Total unique 'subjid' in final_df: {df['subjid'].nunique()}")
    print(f"Total unique 'subjid' in consent: {n_patients_consent}")

    # Save the specific DataFrame for this hospital
    df.to_pickle(hospital_df_path)

# Concatenate all hospital-specific DataFrames after the loop
final_df = pd.concat(dfs, ignore_index=True)
#check if there are duplicates
print("Checking for duplicate patients in subjid...")
print("Dimensions of final_df before removing duplicates: ", final_df.shape)
print(f"Total unique 'subjid' in final_df: {final_df['subjid'].nunique()}")
final_df.drop_duplicates(subset="subjid", keep="first", inplace=True)
print(f"Total unique 'subjid' in final_df: {final_df['subjid'].nunique()}")


# Save the final concatenated DataFrame
#final_df_path = os.path.join(saved_result_path, "final_df.pkl")
final_df_path = os.path.join(saved_result_path, "df_Global_raw.pkl")

final_df.to_pickle(final_df_path)
#save it to csv
final_df.to_csv(os.path.join(saved_result_path, "df_Global_raw.csv"), index=False)

# add Hospital column to the final_df
#merge_patient_dataframes(hospital_names, saved_result_path)


for hospital_name in hospital_names:
    pickle_file_path = os.path.join(saved_result_path, f"df_{hospital_name}.pkl")
    df = pd.read_pickle(pickle_file_path)
    print(f"Total unique 'subjid' in {hospital_name}: {df['subjid'].nunique()}")
    nRow, nCol = df.shape
    print(f"DataFrame dimensions for {hospital_name}: {nRow} rows, {nCol} columns")

print("Loading and previewing global DataFrame...")
pickle_file_path = os.path.join(saved_result_path, "df_Global_raw.pkl")
df = pd.read_pickle(pickle_file_path)
nRow, nCol = df.shape
print(f"Global DataFrame dimensions: {nRow} rows, {nCol} columns")
print("Preview of df Global: ")
print(df.head())


print("Starting general preprocessing steps...")

nan_counts = count_nan_per_column(df)
# print(f"NaN value count per column: {nan_counts}")

total_patients = calculate_total(
    global_path, form_names, hospital_names, calc_type="patients"
)
# print(f"Total patients calculated: {total_patients}")

print("Removing prefixes from specific columns...")
df = remove_hp_prefix(df, "psterm__decod")
print("Adding top 5 symptoms columns...")

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
    global_path, "variables_mapping/mapping_sympthoms.xlsx"
)
HPO_terms_encephalopathy_path = os.path.join(
    global_path, "variables_mapping/HPO_terms_encephalopathy_PL.xlsx"
)
df = update_encephalopathy_column(
    df, mapping_symptoms_path, HPO_terms_encephalopathy_path
)

print("Adding abnormalities columns...")
mapping_abnormalities_path = os.path.join(
    global_path, "variables_mapping/mapping_abnormalities_HPO.xlsx"
)
df = add_abnormalities_cols(df, mapping_abnormalities_path, mapping_symptoms_path)

print("Ordering columns alphabetically...")
df = order_columns_alphabetically(df, ["Hospital", "subjid", "visdat"])


print("New_shape of df before check duplicates", df.shape)

print("Checking for duplicate patients...")
df = check_multiple_patients(df, "subjid", "visdat", saved_result_path)


# print the distribution of missing values for each column
nan_counts = count_nan_per_column(df)
# print(f"NaN value count per column: {nan_counts}")

# read the file "handle_missing.xlsx" and fill the missing values
df = fill_missing_values_imp_var(df)

# print the distribution of missing values for each column after filling the missing values
nan_counts = count_nan_per_column(df)
# print(f"NaN value count per column after filling missing values: {nan_counts}")


#print("Updating 'subjid' by appending 'hospital'...")
#df["subjid"] = df.apply(lambda row: f"{row['subjid']}_{row['Hospital']}", axis=1)
df = df.drop_duplicates(subset=["subjid", "visdat"], keep="last")

# if the column is all nan, remove it
df = df.dropna(axis=1, how="all")

print("Saving preprocessed DataFrame...")
# print the dimension of the df
nRow, nCol = df.shape
print(f"Final DataFrame dimensions: {nRow} rows, {nCol} columns")
df.to_pickle(saved_result_path + "/df_Global_preprocessed.pkl")

print("Creating Excel version of the DataFrame...")
df.to_csv(saved_result_path + "/df_Global_preprocessed.csv", index=False)

print("Converting DataFrame to numerical format...")
df_num = convert_to_numerical(df)
# print the dimension of the df
nRow, nCol = df_num.shape
print(f"Numerical DataFrame dimensions: {nRow} rows, {nCol} columns")
df_num.to_pickle(os.path.join(saved_result_path, "df_num_Global.pkl"))
df_num.to_csv(saved_result_path + "/df_Global_num.csv", index=False)

print("Plotting missing values...")
plot_missing_values(
    df, os.path.join(global_path, "saved_results/distribution_variables","Histogram_MissingValues_df_Global.png" )
)

print("Script execution completed.")
