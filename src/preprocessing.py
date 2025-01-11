import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from itertools import cycle
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from config import global_path
from sklearn import metrics

"""
Functions to do import data and preprocess them
"""


"""IMPORT DATA AND CREATION OF DF"""


def calculate_unique_patient_counts(df):
    unique_patient_counts = df.groupby("Hospital")["subjid"].nunique()
    total_unique_patients = unique_patient_counts.sum()
    return unique_patient_counts, total_unique_patients


def read_important_vars(important_vars_path):
    df_vars = pd.read_excel(important_vars_path)
    return df_vars


# get important variables for a particular form_name
def get_form_vars(form_name, df, common_vars_list):
    vars = [
        var
        for var in list(
            df.loc[(df["form"] == form_name) & (df["reduction_var"] == "Y"), "variable"]
        )
        if var in common_vars_list
    ]
    return vars


# get important variables and important form names
def get_important_variables(file_path):
    df = pd.read_excel(file_path)
    important_vars_df = df[df["reduction_var"] == "Y"]
    important_vars_list = important_vars_df["variable"].tolist()
    form_names = list(dict.fromkeys(important_vars_df["form"]))  # Unique form names
    # Calculate the number of reduced variables
    num_reduced_vars = len(important_vars_list)
    # Calculate the initial number of total variables
    num_total_vars = len(df["variable"])
    # Calculate the percentage of reduced variables
    percentage_reduced = (num_reduced_vars / num_total_vars) * 100
    # Print the results
    # print(f"List of Reduced Variables: {important_vars_list}")
    print(
        f"Number of Reduced Variables from xlsx shared by the Doctors: {num_reduced_vars}"
    )
    # print(f"Percentage of Reduced Variables wrt Initial variables selected: {percentage_reduced:.2f}%")
    return important_vars_list, form_names


def find_common_important_vars(hospital_names, form_names, global_path, df_vars):
    all_hospitals_found_vars = []
    for hospital_name in hospital_names:
        print(f"  Processing hospital_name: {hospital_name}")
        found_vars = set()
        missing_vars_dict = {}
        # Check common variables
        for form_name in form_names:
            # print(f'  Processing form_name: {form_name}')
            # /content/drive/MyDrive/Colab Notebooks/GENOMIT/GENOMIT_Shared/data/Pisa
            try:
                form_path = os.path.join(
                    global_path, "data", hospital_name, f"genomit-{form_name}.csv"
                )
                df_form = pd.read_csv(form_path, sep="\t")

                # Filter variables for the current form where Reduction_var is 'Y'
                filtered_vars = df_vars[
                    (df_vars["form"] == form_name) & (df_vars["reduction_var"] == "Y")
                ]["variable"]

                missing_vars = []
                # if 'pmd' in form_name:
                # print('clindiag__decod' in df_form.columns)
                for var in filtered_vars:
                    if var in df_form.columns:
                        found_vars.add(var)
                    else:
                        missing_vars.append(var)
                        print("columns of the form:", df_form.columns)
                        print(f"  Variable {var} not found in form {form_name}")

                missing_vars_dict[form_name] = missing_vars

            except FileNotFoundError:
                print(f"  File form_name {form_name} not found. Skipping...")
                continue

        # Add the set of found variables for the current hospital to the list
        all_hospitals_found_vars.append(found_vars)
        # Print final lists and counts of found and missing variables for each hospital
        # print(f"Found Variables for {hospital_name}: {found_vars}")
        print(f"Number of Found Variables for {hospital_name}: {len(found_vars)}")
    common_vars = set.intersection(*all_hospitals_found_vars)
    common_vars_list = list(common_vars)
    print(f"Number of Common Variables: {len(common_vars)}")
    return common_vars_list


def merge_patient_dataframes(hospital_names, saved_result_path):
    dfs = [
        pd.read_pickle(os.path.join(saved_result_path, "df", f"df_{hospital_name}.pkl"))
        for hospital_name in hospital_names
    ]
    df = pd.concat(dfs, keys=hospital_names, names=["Hospital", "RowID"]).reset_index(
        level="Hospital"
    )
    df.to_pickle(os.path.join(saved_result_path, "df", "df_Global.pkl"))
    nRow, nCol = df.shape
    print(f"There are {nRow} rows and {nCol} columns in the df_Global")


"""PRIORITARIZATION"""


# Function to process 'gendna' column: 6 AND 8 AS
def process_gendna(df):
    df["epiphen"] = 0
    unique_subjids = df["subjid"].unique()

    for subj_id in unique_subjids:
        subj_df = df[df["subjid"] == subj_id]
        gendna_values = subj_df["gendna"].dropna().unique()

        if 6 in gendna_values or 8 in gendna_values:
            df.loc[df["subjid"] == subj_id, "epiphen"] = 1

        if 4 in gendna_values or 5 in gendna_values or 7 in gendna_values:
            gendna = [val for val in gendna_values if val in [4, 5, 7]]
            df.loc[df["subjid"] == subj_id, "gendna"] = gendna[
                0
            ]  # Set gendna to the first value in the list

        if len(gendna_values) > 1:
            print(
                f"Duplicates found for 'gendna' in subject ID {subj_id}: {gendna_values}"
            )

    count_non_zero_epiphen = (df["epiphen"] != 0).sum()
    # print("Count of 'epiphen' values different from zero:", count_non_zero_epiphen)

    return df


# Function to process 'psterm__decod' column
def process_psterm_deco(df):
    # Group by 'subjid' and aggregate unique 'psterm__decod' values as a list, excluding NaNs
    grouped = df.groupby("subjid")["psterm__decod"].apply(
        lambda x: list(set(x.dropna()))
    )

    # Find the maximum number of unique non-NaN 'psterm__decod' values for a single patient
    max_values = 26  # grouped.apply(len).max()

    # Create new columns for each 'psterm__decod' value
    for i in range(max_values):
        # df[f'psterm__decod_{i}'] = np.nan
        df.loc[:, f"psterm__decod_{i}"] = np.nan

    # Fill the new columns sequentially
    for subjid, values in grouped.items():
        for i, value in enumerate(values):
            df.loc[df["subjid"] == subjid, f"psterm__decod_{i}"] = value
            # print('subj, values', subjid, values)

    # Print the max number of unique values
    # print("Max number of unique 'psterm__decod' values for a single patient (excluding NaNs):", max_values)

    # Print the resulting DataFrame
    return df


# Function to process tissue exam columns NOT used at the moment
def process_tissue_exams(df, exam_columns):
    unique_subjids = df["subjid"].unique()
    for subj_id in unique_subjids:
        subj_df = df[df["subjid"] == subj_id]
        for col in exam_columns:
            latest_exam = subj_df[subj_df[col] == subj_df[col].max()]
            if len(latest_exam) > 1:
                # print(f"Multiple exams found for patient {subj_id} at the same date.")
                last_row = subj_df.iloc[-1]  # Get the last row for the patient
                if last_row[exam_columns].isna().all():
                    # If all values in exam_columns are NaN, look for a non-NaN row previously
                    prev_non_nan_row = subj_df[subj_df[exam_columns].notna()].iloc[-1]
                    last_row = prev_non_nan_row
                for col in exam_columns:
                    subj_df.loc[subj_df.index, col] = last_row[col]
                df.loc[df["subjid"] == subj_id, exam_columns] = subj_df[
                    exam_columns
                ].values
    return df


# Function to process 'pimgtype' column
def process_pimgtypelog(df):
    # Step 1: Find unique non-NaN values in 'pimgtype' and order them alphabetically
    unique_pimgtypes = sorted(df["pimgtype"].dropna().unique())

    # Print the unique 'pimgtype' values and their corresponding columns
    # print('Unique pimgtype values:', unique_pimgtypes)

    # Step 2: Create a new column for each unique 'pimgtype' value
    for pimgtype in unique_pimgtypes:
        df[f"pimgtype_{pimgtype}"] = np.nan

    # Step 3: Set the columns to 1 for each patient if 'pimgtype' matches a column name
    for pimgtype in unique_pimgtypes:
        column_name = f"pimgtype_{pimgtype}"
        df.loc[df["pimgtype"] == pimgtype, column_name] = 1
    return df


def process_pimgtype(df):
    # Step 4: Set 'sll' to 0 if 'mrisll' has all values equal to 998 or NaN
    df.loc[(df["mrisll"].isna() | (df["mrisll"] == 998)), "sll"] = 0

    # Step 2: Set 'sll' to 1 if 'mrisll' has at least one value of 1 and not 0
    df.loc[(df["mrisll"] == 1) & (df["mrisll"] != 0), "sll"] = 1

    # Step 3: Set 'sll' to 2 if 'mrisll' has at least one value of 0 and not 1
    df.loc[(df["mrisll"] == 0) & (df["mrisll"] != 1), "sll"] = 2

    # Step 4: Set 'sll' to 3 if 'mrisll' has at least one 1 followed by a 0 in the following rows of the same patient
    df["sll"] = (
        df.groupby("subjid")["mrisll"]
        .transform(
            lambda x: (
                (x.eq(1) & x.shift(-1).eq(0) & x.shift(-1).notna())
                | (x.eq(3) & x.shift(-1).eq(0) & x.shift(-1).notna())
            )
        )
        .astype(int)
    )

    count_non_zero_sll = (df["sll"] != 0).sum()
    # print("Count of 'sll' values different from zero:", count_non_zero_sll)

    return df


# Function to process 'pimgres' and 'mrisll' columns
def process_pimgres_mrisll(df):
    df["sll"] = (df["pimgres"] != 3) & (df["mrisll"] == 1)
    return df


def process_form(subjid_df, form_name, vars):
    # prioritarization geneticresults_log
    if "geneticresultlog" in form_name:
        # subjid_df['epiphen'] = 0
        vars = vars + ["epiphen"]
        subjid_df = process_gendna(subjid_df)
        df_copy = subjid_df.copy()  # to transforn it in numerical
        # print('processed df for the subject with epiphenomena')
        # print(subjid_df)
    # prioritarization pdm_sympthoms
    if "pmdsymptoms" in form_name:
        list_sym = []  # to add the syms to vars
        for i in range(max_symps):
            list_sym.append(f"psterm__decod_{i}")
        # print(list_sym)
        vars = vars + list_sym
        subjid_df = process_psterm_deco(subjid_df)
        df_copy = subjid_df.copy()  # to transforn it in numerical
        # print('processed df for the subject with added psterm__decod')
        # print(subjid_df)
        # print(subjid_df.columns)
    # prioritarization tissue_lab_results
    if (
        "tissuelabresultlog_not_valid" in form_name
    ):  # the function work but i am not using it.
        exam_columns = [
            "amc",
            "ckval",
            "dacco",
            "dcd",
            "hba1c",
            "lacval",
            "md",
            "nc",
            "rrf",
        ]
        subjid_df = process_tissue_exams(subjid_df, exam_columns)
        df_copy = subjid_df.copy()  # to transforn it in numerical
        print("processed df for the subject with added tissue_exams")
        print(subjid_df)
    # prioritarization pmdimagingresultlog
    if "pmdimagingresultlog" in form_name:
        unique_pimgtypes_subj = sorted(subjid_df["pimgtype"].dropna().unique())
        list_img = []  # to add the syms to vars
        for pimgtype in unique_pimgtypes_subj:
            list_img.append(f"pimgtype_{pimgtype}")
        # print(list_img)
        vars = vars + list_img
        # print('vars to be added:',vars)
        # add pimgtype to the df for that subject
        subjid_df = process_pimgtypelog(subjid_df)
        # print('processed df for the subject with added img')
        # print(subjid_df)
        # print(subjid_df.columns)
        df_copy = subjid_df.copy()  # to transforn it in numerical
    # prioritarization pmdimagingresultlog
    if form_name == "pmdimagingresult":
        # print('add SLL column!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        vars = vars + ["sll"]
        subjid_df = process_pimgtype(subjid_df)
        df_copy = subjid_df.copy()  # to transforn it in numerical
        # print('processed df for the subject with sll')
        # print(subjid_df)
    return [subjid_df, vars]


"""PREPROCESSING"""


def count_nan_per_column(df):
    # print(df.isnull().sum())
    return df.isnull().sum()


# calculate the number of total variables across the different forms
def get_unique_columns(hospital_names, global_path):
    unique_columns = set()

    # Iterate over hospitals
    for hospital_name in hospital_names:
        # Form path for the hospital
        hospital_path = os.path.join(global_path, "data", hospital_name)

        # Check if the hospital folder exists
        if os.path.exists(hospital_path):
            # List all files in the hospital folder
            files = os.listdir(hospital_path)

            # Iterate over files to extract form names
            for file in files:
                # Extract form name from file name
                form_name, file_extension = os.path.splitext(file)

                # Check if the file is a valid form file (xlsx or csv)

                form_path = os.path.join(hospital_path, file)

                # Read form
                if "genomit-" in form_path:
                    df_form = pd.read_csv(form_path, sep="\t")

                    # Get unique column names for the current form
                    unique_columns.update(df_form.columns)

    # Return the unique column names
    return unique_columns


# calc total number of patients for each hospital
def calculate_total(global_path, form_names, hospital_names, calc_type="patients"):
    total_list_hospitals = []
    for hospital_name in hospital_names:
        total_list_hospital = []
        for form_name in form_names:
            try:

                form_path = os.path.join(
                    global_path, "data", hospital_name, f"genomit-{form_name}.csv"
                )
                df_form = pd.read_csv(form_path, sep="\t")

                if calc_type == "patients":
                    items = df_form["subjid"].values
                elif calc_type == "visits":
                    items = df_form["visdat"].values
                    items = items.astype("datetime64[s]").astype(np.float64)
                else:
                    print("Invalid calc_type. Use either 'patients' or 'visits'.")
                    return

                total_list_hospital = np.unique(
                    np.concatenate((total_list_hospital, items), axis=0)
                )
            except:
                continue

        print(
            f"Total number of unique {calc_type} for {hospital_name}: ",
            len(total_list_hospital),
        )
        total_list_hospitals.append(len(total_list_hospital))

    total_all_hospitals = sum(total_list_hospitals)
    print(
        f"Total number of unique {calc_type} for all hospitals: ", total_all_hospitals
    )

    return total_list_hospitals, total_all_hospitals


def remove_hp_prefix(df, column_prefix):
    columns_to_process = [col for col in df.columns if column_prefix in col]
    print("Removing Hp from these cols: ", columns_to_process)

    for col in columns_to_process:
        df[col] = df[col].apply(
            lambda x: x.replace("HP:", "") if isinstance(x, str) and "HP:" in x else x
        )

    return df


def uniformate_sex(df, column_name):
    df[column_name] = df[column_name].replace({"m": 0, "f": 1})
    sex_distribution = df["sex"].value_counts(normalize=True) * 100

    # Print the distribution
    print("Distribution of the 'sex' column:")
    print(sex_distribution)
    # Assuming 'm' corresponds to 0 and 'f' corresponds to 1
    percentage_male = sex_distribution[0] if 0 in sex_distribution.index else 0
    percentage_female = sex_distribution[1] if 1 in sex_distribution.index else 0

    print(f"Percentage of males: {percentage_male:.2f}%")
    print(f"Percentage of females: {percentage_female:.2f}%")

    return df


def get_oldest_visits(df, date_column, group_column, saved_result_path):
    oldest_visit_indices = df.groupby(group_column)[date_column].idxmin()
    df_old = df.loc[oldest_visit_indices]
    df_old.reset_index(drop=True, inplace=True)
    df_old = df_old.drop_duplicates(
        subset=group_column, keep="first"
    )  # attention to modify duplicants
    # Save the processed DataFrame to pickle
    df.to_pickle(os.path.join(saved_result_path, "df_old_Global.pkl"))
    return df_old


def process_datetime_column(df, column_name):
    # print(df[column_name])
    # print(f'Before processing {column_name}:', df[column_name].isnull().sum())

    # Convert 'column_name' column to datetime if it's not already
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    # Convert datetime64 to string with only year-month-day
    df[column_name] = df[column_name].dt.strftime("%Y-%m-%d")

    # print(f'After processing {column_name}:', df[column_name].isnull().sum())


def process_datetime_column_onset(df, column_name):
    # Convert 'column_name' column to datetime if it's not already
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    # Convert datetime64 to string with only year-month-day
    df[column_name] = df[column_name].dt.strftime("%Y-%m-%d")


def order_columns_alphabetically(df, initial_columns):
    """
    Order the columns in a DataFrame alphabetically, with specific columns at the beginning.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be processed.
    - initial_columns (list): List of column names to keep at the beginning.

    Returns:
    - pd.DataFrame: DataFrame with columns ordered as specified.
    """
    # Ensure all initial columns are present in the DataFrame
    missing_columns = set(initial_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Reorder columns
    df = df[initial_columns + sorted(df.columns.difference(initial_columns))]

    return df


def add_top5_symptoms_columns(
    df_preprocessed_global,
    hospital_names,
    consent_form_name,
    symptoms_form_name,
    pmd_form_name,
):
    top5_columns = [
        f"symp_on_{i}" for i in range(1, 6)
    ]  # Define the column names outside the loop
    all_symptoms_dfs = []  # List to store symptoms_df for each hospital

    for hospital_name in hospital_names:
        # Determine file extension based on hospital_name
        file_extension = "csv"

        # Consent form, Symptoms form, and PMD form
        consent_path = os.path.join(
            global_path,
            "data",
            hospital_name,
            f"genomit-{consent_form_name}.{file_extension}",
        )
        symptoms_path = os.path.join(
            global_path,
            "data",
            hospital_name,
            f"genomit-{symptoms_form_name}.{file_extension}",
        )
        pmd_path = os.path.join(
            global_path,
            "data",
            hospital_name,
            f"genomit-{pmd_form_name}.{file_extension}",
        )

        consent_df = pd.read_csv(consent_path, sep="\t")
        symptoms_df = pd.read_csv(symptoms_path, sep="\t")
        pmd_df = pd.read_csv(pmd_path, sep="\t")

        # Merge the two forms on subjid
        merged_df = pd.merge(
            symptoms_df, consent_df[["subjid", "brthdat__c"]], on="subjid", how="left"
        )

        # Merge the 'pmd' form on subjid to get 'aao'
        merged_df = pd.merge(
            merged_df, pmd_df[["subjid", "aao"]], on="subjid", how="left"
        )

        # Process date columns
        process_datetime_column(merged_df, "psstdat__c")
        process_datetime_column(merged_df, "brthdat__c")

        # Calculate age at symptoms (AAS) in years
        merged_df["aas"] = (
            pd.to_datetime(merged_df["psstdat__c"], errors="coerce")
            - pd.to_datetime(merged_df["brthdat__c"], errors="coerce")
        ).dt.days / 365.25

        # Calculate the difference between AAS and AAO for each patient
        merged_df["aas_aao_diff"] = merged_df["aao"] - merged_df["aas"]

        # For each patient, select the top 5 symptoms with the lowest absolute values of 'aas_aao_diff'
        top5_symptoms = (
            merged_df.groupby("subjid", as_index=False)
            .apply(lambda group: group.nsmallest(5, "aas_aao_diff"))
            .reset_index(drop=True)
        )

        # Ensure that the selected symptoms are unique for each patient
        top5_symptoms = (
            top5_symptoms.groupby("subjid")["psterm__decod"]
            .apply(lambda x: x.tolist())
            .reset_index(name="top5_symptoms")
        )

        symptoms_df = pd.DataFrame(
            top5_symptoms["top5_symptoms"].tolist(), index=top5_symptoms["subjid"]
        ).fillna(0)

        # Rename the columns to symp_on_n
        symptoms_df.columns = top5_columns

        # print('symphtoms_df of :', hospital_name)
        # print(symptoms_df)
        all_symptoms_dfs.append(symptoms_df)  # Store symptoms_df in the list

        # print('all_symptoms_dfs: ', all_symptoms_dfs)

        # print(f'done for {hospital_name}')

    # Concatenate all symptoms_df DataFrames
    symptoms_df_global = pd.concat(all_symptoms_dfs, axis=0)
    symptoms_df_global.drop_duplicates(inplace=True)

    df_preprocessed_global = pd.merge(
        df_preprocessed_global,
        symptoms_df_global[list(top5_columns)],
        left_on="subjid",
        right_index=True,
        how="left",
    )

    # Select only one row with the fewest missing values for each unique combination of 'Hospital', 'subjid', and 'visdat' if duplicates exist
    duplicated_indices = df_preprocessed_global.duplicated(
        subset=["Hospital", "subjid", "visdat"], keep=False
    )
    df_preprocessed_global_duplicated = df_preprocessed_global[duplicated_indices]
    df_preprocessed_global_unique = df_preprocessed_global[~duplicated_indices]
    if not df_preprocessed_global_duplicated.empty:
        print(df_preprocessed_global_duplicated)
        # Keep only one row randomly
        df_preprocessed_global_duplicated = (
            df_preprocessed_global_duplicated.groupby(["Hospital", "subjid", "visdat"])
            .apply(lambda x: x.sample(n=1))
            .reset_index(drop=True)
        )
        df_preprocessed_global = pd.concat(
            [df_preprocessed_global_unique, df_preprocessed_global_duplicated],
            ignore_index=True,
        )
        print(df_preprocessed_global_duplicated)

    return df_preprocessed_global


def update_encephalopathy_column(
    df, mapping_symptoms_path, HPO_terms_encephalopathy_path
):

    # Read mapping_abnormalities_HPO.xlsx
    mapping_enc = pd.read_excel(HPO_terms_encephalopathy_path)

    # Read mapping_symptoms.xlsx
    mapping_symptoms = pd.read_excel(mapping_symptoms_path)

    # Extract terms from the Encephalopathy column of the mapping_enc dataframe
    encephalopathy_terms = mapping_enc["Encephalopathy"].tolist()

    # Match the terms from mapping_enc with the Corresponding_psterm_decod in mapping_symptoms
    matched_HPO_terms_enc = mapping_symptoms[
        mapping_symptoms["psterm__modify"].isin(encephalopathy_terms)
    ]["Corresponding_psterm_decod"].tolist()

    # Print the list of Corresponding_psterm_decod terms that match the terms in mapping_enc
    # print(matched_HPO_terms_enc)

    # Create a new column 'encephalopathy' with default value 0
    df["encephalopathy"] = 0

    # Iterate over rows
    for index, row in df.iterrows():
        # Extract columns with names containing 'psterm__decod'
        psterm_decod_columns = [col for col in df.columns if "psterm__decod" in col]

        # Create a list of non-NaN values from these columns
        values = row[psterm_decod_columns].dropna().tolist()

        # Check if there's at least one term in the unique_psterm_decod_values list
        if any(term in values for term in matched_HPO_terms_enc):
            # print('Encephalopathy')
            # Update the 'encephalopathy' column for this row to 1
            df.at[index, "encephalopathy"] = 1

    encephalopathy_distribution = df["encephalopathy"].value_counts()
    print("encephalopathy_distribution: ", encephalopathy_distribution)

    return df


# function to add ebnormalities columns
def add_abnormalities_cols(df, mapping_abnormalities_path, mapping_symptoms_path):
    # Read the main DataFrame

    # Read mapping_abnormalities_HPO.xlsx
    mapping_abnormalities = pd.read_excel(mapping_abnormalities_path, index_col=0)

    # Read mapping_symptoms.xlsx
    mapping_symptoms = pd.read_excel(mapping_symptoms_path)

    # Define the column mapping
    column_mapping = {
        "Behavioral/psychiatric abnormality": "beh_psy_abn",
        "Cardiovascular system": "card_abn",
        "Digestive system": "diges_abn",
        "Ear or voice": "ear_voice_abn",
        "Eye": "eye_abn",
        "Gait Disturbance": "gait_abn",
        "Prenatal development or birth/growth abnormality": "natal_growth_abn",
        "Genito urinaty system or breast ": "genit_breast_abn",
        "Head or neck": "head_neck_abn",
        "Internal / endocrine system or blood": "internal_abn",
        "Muskulo-skeletal system": "musc_sk_abn",
        "Nervous system": "nerv_abn",
        "Other abnormality": "other_abn",
        "Respiratory system": "resp_abn",
        "Connective tissue / skin": "conn_skin_abn",
    }

    # Rename the columns
    mapping_abnormalities.rename(columns=column_mapping, inplace=True)

    # Iterate over columns in mapping_abnormalities
    if (
        "psterm__modify" in mapping_symptoms.columns
        and "Corresponding_psterm_decod" in mapping_symptoms.columns
    ):
        # Create a dictionary mapping psterm__modify to Corresponding_psterm_decod
        modify_to_decod_mapping = mapping_symptoms.set_index("psterm__modify")[
            "Corresponding_psterm_decod"
        ].to_dict()

        # Replace values in the column with Corresponding_psterm_decod
        mapping_abnormalities = mapping_abnormalities.applymap(
            lambda x: modify_to_decod_mapping.get(x, x)
        )

    # Add new columns to df based on mapping_abnormalities
    for column in mapping_abnormalities.columns:
        df[column] = None  # Initialize the new columns

    subjid_list = df["subjid"].tolist()
    patients_symptoms = []
    for index, row in df.iterrows():
        symptoms = set(
            row[col]
            for col in df.columns
            if "psterm__decod" in col and not pd.isna(row[col])
        )
        patients_symptoms.append(list(symptoms))

    subjid_list = df["subjid"].tolist()
    patients_symptoms_mapping = {}

    for index, row in df.iterrows():
        subjid = row["subjid"]
        symptoms = set(
            row[col]
            for col in df.columns
            if "psterm__decod" in col and not pd.isna(row[col])
        )
        patients_symptoms_mapping[subjid] = list(symptoms)

    # Iterate through the list of symptoms and update mapping_abnormalities
    # for index, symptoms in enumerate(patients_symptoms):
    for subjid, symptoms in patients_symptoms_mapping.items():
        for symptom in symptoms:
            for col in mapping_abnormalities.columns:
                hpo_values = mapping_abnormalities[col].values
                hpo_values = hpo_values[~pd.isna(hpo_values)]

                stripped_symptom = str(int(symptom))
                stripped_hpo_values = [str(int(val)) for val in hpo_values]

                index_of_subjid = df.loc[df["subjid"] == subjid].index[0]

                if stripped_symptom in stripped_hpo_values:
                    df.at[index_of_subjid, col] = 1

    # Print the distribution of each column in the final DataFrame
    for col in mapping_abnormalities.columns:
        print(f"Distribution for column {col}:")
        print(df[col].value_counts())
        print("\n")

    return df


def convert_to_numerical(df):  # ,saved_result_path, hospital_name): #mantaining nans

    numerical_cols = []
    non_numerical_cols = []

    # Check each column's data type
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            non_numerical_cols.append(col)

    # Print lists
    # print(f"Numerical Columns: {numerical_cols}")
    # print('Amount of numerical Columns: ', len(numerical_cols))
    # print(len(numerical_cols))
    # print(f"Non-Numerical Columns: {non_numerical_cols}")
    # print('Number of Non-Numerical Columns: ', len(non_numerical_cols))

    # do not convert subjid column
    # Check if the 'subjid' column is numerical or not and print a message
    if "subjid" in df.columns:
        if pd.api.types.is_numeric_dtype(df["subjid"]):
            print("The 'subjid' column is numerical.")
        else:
            print("The 'subjid' column is not numerical.")
        # Remove 'subjid' from non_numerical_cols if it exists
        if "subjid" in non_numerical_cols:
            non_numerical_cols.remove("subjid")
    else:
        print("The 'subjid' column does not exist in the DataFrame.")

    df_copy = df.copy()
    for col in non_numerical_cols:
        if col in df.columns:
            df_copy[col] = df_copy[col].apply(lambda x: str(x) if not pd.isna(x) else x)
            label_encoder = LabelEncoder()
            df_copy[col] = label_encoder.fit_transform(df_copy[col])

    nan_mask = df.isna()
    df_copy = df_copy.mask(
        nan_mask, np.nan
    )  # Use the mask to replace values in df_old_num with NaNs

    return df_copy


# PRIORITARIZATION#
# Function to process 'gendna' column: 6 AND 8 AS
def process_gendna(df):
    df["epiphen"] = 0
    unique_subjids = df["subjid"].unique()

    for subj_id in unique_subjids:
        subj_df = df[df["subjid"] == subj_id]
        gendna_values = subj_df["gendna"].dropna().unique()

        if 6 in gendna_values or 8 in gendna_values:
            df.loc[df["subjid"] == subj_id, "epiphen"] = 1

        if 4 in gendna_values or 5 in gendna_values or 7 in gendna_values:
            gendna = [val for val in gendna_values if val in [4, 5, 7]]
            df.loc[df["subjid"] == subj_id, "gendna"] = gendna[
                0
            ]  # Set gendna to the first value in the list

        if len(gendna_values) > 1:
            print(
                f"Duplicates found for 'gendna' in subject ID {subj_id}: {gendna_values}"
            )

    count_non_zero_epiphen = (df["epiphen"] != 0).sum()
    # print("Count of 'epiphen' values different from zero:", count_non_zero_epiphen)

    return df


def preprocess_gendna_data(df):
    """Preprocess 'gendna' column dividing in two classes and printing the distribution."""
    # Drop NaN values from 'gendna' column
    df_non_nan = df.dropna(subset=["gendna"])
    df_non_nan = df_non_nan[
        df_non_nan["gendna"] != 1
    ]  # Remove rows where 'gendna' is equal to 1

    # Create 'nDNA' and 'mtDNA' classes
    df_non_nan["nDNA"] = df_non_nan["gendna"].apply(
        lambda x: "nDNA" if x in [4, 6, 8] else None
    )
    df_non_nan["mtDNA"] = df_non_nan["gendna"].apply(
        lambda x: "mtDNA" if x in [5, 7] else None
    )

    # Print the number of samples and the percentage of each class in non-NaN rows
    for class_name in ["nDNA", "mtDNA"]:
        class_samples = df_non_nan[class_name].dropna()
        total_samples = len(class_samples)
        percentage = total_samples / len(df_non_nan) * 100
        print(f"Number of samples in {class_name}: {total_samples} ({percentage:.2f}%)")

    # Combine 'nDNA' and 'mtDNA' classes into 'gendna_type' column
    df_non_nan["gendna_type"] = df_non_nan.apply(
        lambda row: row["nDNA"] if row["nDNA"] is not None else row["mtDNA"], axis=1
    )

    print("\nDistribution of 'gendna_type':")
    print(df_non_nan["gendna_type"].value_counts())

    return df_non_nan


# Function to process 'psterm__decod' column
def process_psterm_deco(df):
    # Group by 'subjid' and aggregate unique 'psterm__decod' values as a list, excluding NaNs
    grouped = df.groupby("subjid")["psterm__decod"].apply(
        lambda x: list(set(x.dropna()))
    )

    # Find the maximum number of unique non-NaN 'psterm__decod' values for a single patient
    max_values = 26  # grouped.apply(len).max()

    # Create new columns for each 'psterm__decod' value
    for i in range(max_values):
        df[f"psterm__decod_{i}"] = np.nan

    # Fill the new columns sequentially
    for subjid, values in grouped.items():
        for i, value in enumerate(values):
            df.loc[df["subjid"] == subjid, f"psterm__decod_{i}"] = value
            # print('subj, values', subjid, values)

    # Print the max number of unique values
    # print("Max number of unique 'psterm__decod' values for a single patient (excluding NaNs):", max_values)

    # Print the resulting DataFrame
    return df


# Function to process tissue exam columns NOT used at the moment
def process_tissue_exams(df, exam_columns):
    unique_subjids = df["subjid"].unique()
    for subj_id in unique_subjids:
        subj_df = df[df["subjid"] == subj_id]
        for col in exam_columns:
            latest_exam = subj_df[subj_df[col] == subj_df[col].max()]
            if len(latest_exam) > 1:
                # print(f"Multiple exams found for patient {subj_id} at the same date.")
                last_row = subj_df.iloc[-1]  # Get the last row for the patient
                if last_row[exam_columns].isna().all():
                    # If all values in exam_columns are NaN, look for a non-NaN row previously
                    prev_non_nan_row = subj_df[subj_df[exam_columns].notna()].iloc[-1]
                    last_row = prev_non_nan_row
                for col in exam_columns:
                    subj_df.loc[subj_df.index, col] = last_row[col]
                df.loc[df["subjid"] == subj_id, exam_columns] = subj_df[
                    exam_columns
                ].values
    return df


# Function to process 'pimgtype' column
def process_pimgtypelog(df):
    # Step 1: Find unique non-NaN values in 'pimgtype' and order them alphabetically
    unique_pimgtypes = sorted(df["pimgtype"].dropna().unique())

    # Print the unique 'pimgtype' values and their corresponding columns
    # print('Unique pimgtype values:', unique_pimgtypes)

    # Step 2: Create a new column for each unique 'pimgtype' value
    for pimgtype in unique_pimgtypes:
        df[f"pimgtype_{pimgtype}"] = np.nan

    # Step 3: Set the columns to 1 for each patient if 'pimgtype' matches a column name
    for pimgtype in unique_pimgtypes:
        column_name = f"pimgtype_{pimgtype}"
        df.loc[df["pimgtype"] == pimgtype, column_name] = 1
    return df


def process_pimgtype(df):
    # Step 4: Set 'sll' to 0 if 'mrisll' has all values equal to 998 or NaN
    df.loc[(df["mrisll"].isna() | (df["mrisll"] == 998)), "sll"] = 0

    # Step 2: Set 'sll' to 1 if 'mrisll' has at least one value of 1 and not 0
    df.loc[(df["mrisll"] == 1) & (df["mrisll"] != 0), "sll"] = 1

    # Step 3: Set 'sll' to 2 if 'mrisll' has at least one value of 0 and not 1
    df.loc[(df["mrisll"] == 0) & (df["mrisll"] != 1), "sll"] = 2

    # Step 4: Set 'sll' to 3 if 'mrisll' has at least one 1 followed by a 0 in the following rows of the same patient
    df["sll"] = (
        df.groupby("subjid")["mrisll"]
        .transform(
            lambda x: (
                (x.eq(1) & x.shift(-1).eq(0) & x.shift(-1).notna())
                | (x.eq(3) & x.shift(-1).eq(0) & x.shift(-1).notna())
            )
        )
        .astype(int)
    )

    count_non_zero_sll = (df["sll"] != 0).sum()
    # print("Count of 'sll' values different from zero:", count_non_zero_sll)

    return df


# Function to process 'pimgres' and 'mrisll' columns
def process_pimgres_mrisll(df):
    df["sll"] = (df["pimgres"] != 3) & (df["mrisll"] == 1)
    return df


"""

    # Step 5: Handle multiple exams for the same date and patient
    unique_subjids = df['subjid'].unique()

    for subj_id in unique_subjids:
        subj_df = df[df['subjid'] == subj_id]
        exam_columns = ['pimgtype']
        for col in exam_columns:
            latest_exam = subj_df[subj_df[col] == subj_df[col].max()]
            if len(latest_exam) > 1:
                last_row = subj_df.iloc[-1]  # Get the last row for the patient
                if last_row[exam_columns].isna().all():
                    prev_non_nan_row = subj_df[subj_df[exam_columns].notna()].iloc[-1]
                    last_row = prev_non_nan_row
                for col in exam_columns:
                    subj_df.loc[subj_df.index, col] = last_row[col]
                df.loc[df['subjid'] == subj_id, exam_columns] = subj_df[exam_columns].values

    return df

# Define a function to prioritize 'gendna' values
def prioritize_gendna(df):
    # Create a dictionary to define the priority of gendna values
    gendna_priority = {4: 1, 5: 2, 7: 3, 2: 4, 8: 5, 6: 6}
    
    # Calculate the number of occurrences of each subject ID
    subject_id_counts = df['subjid'].value_counts()
    
    # Iterate over the DataFrame to prioritize rows based on gendna values
    selected_rows = []
    duplicates = []

    for subjid, group in df.groupby('subjid'):
        if len(group) == 1:
            selected_rows.append(group)
        else:
            priority_group = group[group['gendna'].isin(gendna_priority)]
            if not priority_group.empty:
                max_priority_value = min(priority_group['gendna'], key=lambda x: gendna_priority[x])
                selected_rows.append(priority_group[priority_group['gendna'] == max_priority_value])
                if len(priority_group['gendna'].unique()) > 1:
                    duplicates.append((subjid, priority_group['gendna'].unique()))
            else:
                # Handle cases with gendna values not in [4, 5, 7, 2, 8, 6]
                selected_rows.append(group)
                unknown_values = group[~group['gendna'].isin(gendna_priority.keys())]
                if not unknown_values.empty:
                    print(f"Unknown gendna value(s) for subject ID {subjid}: {unknown_values['gendna'].unique()}")
    
    # Concatenate the selected rows back into a DataFrame
    selected_rows_df = pd.concat(selected_rows)
    if duplicates:
        for subjid, values in duplicates:
            print(f"Duplicates found for 'gendna' in subject ID {subjid}. Rows maintained for duplicates: {values}")
    
    # Print the dimensions of the resulting DataFrame
    nRow, nCol = selected_rows_df.shape
    print(f'There are {nRow} rows and {nCol} columns in the df_Global after gendna processing')
    
    return selected_rows_df


# Define a function to keep rows with the most recent 'visdat' for each 'pimgtype' value
def prioritize_pimgtype(df):
    df['visdat'] = pd.to_datetime(df['visdat'])
    df = df.sort_values(['subjid', 'pimgtype', 'visdat'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['subjid', 'pimgtype'], keep='first')
    duplicates = df[df.duplicated(subset=['subjid', 'pimgtype'], keep=False)]
    if not duplicates.empty:
        print(f"Duplicates found for 'pimgtype'. Number of duplicates: {len(duplicates)}")
    return df

# Define a function to keep rows with the most recent 'visdat' for each patient
def prioritize_tissueandlab(df):
    df['visdat'] = pd.to_datetime(df['visdat'])
    df = df.sort_values(['subjid', 'visdat'], ascending=[True, False])
    df = df.drop_duplicates(subset=['subjid'], keep='first')
    duplicates = df[df.duplicated(subset=['subjid'], keep=False)]
    if not duplicates.empty:
        print(f"Duplicates found for 'tissueandlab'. Number of duplicates: {len(duplicates)}")
    return df
"""
