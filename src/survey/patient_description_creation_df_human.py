import os
import pandas as pd
import numpy as np

from datetime import datetime

import time
import sys
import pandas as pd
import os
import re


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from preprocessing import add_abnormalities_cols
from processing import fill_missing_values


# Get the current script's file name
current_file = os.path.basename(__file__)
print("the current file is: ", current_file)
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(script_directory)
GLOBAL_PATH = os.path.dirname(parent_path)


saved_result_path = os.path.join(GLOBAL_PATH, "saved_results")
SURVEY_PATH = os.path.join(saved_result_path, "survey")
#BEST_PATH = os.path.join(saved_result_path, "classifiers_results/best_model")


#df_path = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
df_path=os.path.join(saved_result_path, "df", "df_no_symp.csv") 
#df_test_path = os.path.join(SURVEY_PATH, "df_test_best.csv")
df_test_path=  os.path.join(saved_result_path, "df", "df_no_symp.csv") 
important_vars_path = os.path.join(
    GLOBAL_PATH, "variables_mapping", "important_variables_huma.xlsx"
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
    df_test = pd.read_csv(df_test_path)
    print(f"Loaded DataFrame: {df_test_path}")
except FileNotFoundError:
    print(f"Error: File not found at {df_test_path}")
    exit()

nRow, nCol = df_test.shape
print(f"The DataFrame 'df_test' contains {nRow} rows and {nCol} columns.")



"""
FILTER AND ANALYZE COLUMNS
"""
#print the cols of the df_test
print("Columns of the df_test:", df_test.columns)

input = input("Do you want to filter the columns of the df_test? (y/n): ")
# Filter `df_raw` by `subject_id_test` and display new shape
#subject_id_test = df_test["subjid"].tolist()

# Filter `df_raw` by `subject_id_test` and display new shape
subject_id_test = df_test[df_test["test"] == 1]["subjid"].values

#print the len of the subject_id_test
print(f"Number of subjects in test set: {len(subject_id_test)}")

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
"""
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
"""

# load the selected features selected_features.txt
'''
selected_features_path= os.path.join(BEST_PATH, "selected_features.txt")
with open(selected_features_path, "r") as file:
    selected_features = file.read().splitlines()
    print("Selected features:", selected_features)'
'''
#TODO insert automatically them
selected_features=['ear_voice_abn', 'genit_breast_abn', 'sex', 'card_abn', 'alcohol', 'internal_abn', 'eye_abn', 'ecgh', 'resp', 'hba1c', 'ear', 'lac', 'echo', 'ecg', 'oct', 'bmi', 'hmri', 'pcgntn', 'natal_growth_abn', 'smoking', 'symp_on_2', 'vsupp', 'eeg', 'symp_on_4', 'mhterm', 'pimgres', 'va', 'ck', 'aao', 'conn_skin_abn', 'oa', 'nerv_abn', 'emg', 'diges_abn', 'wml', 'symp_on_1', 'musc_sk_abn', 'symp_on_5', 'resp_abn', 'fhx', 'other_abn', 'hisc', 'pssev', 'mrisll', 'cc', 'gait_abn', 'bgc', 'fo', 'symp_on_3', 'le']
# add subjid to the selected features
selected_features.append("subjid")

# Drop columns not in selected features
columns_to_drop = [col for col in df_raw.columns if col not in selected_features]
df = df_raw.drop(columns=columns_to_drop, errors="ignore")



#["subjid"].tolist()
df_raw = df_raw[df_raw["subjid"].isin(subject_id_test)]
print(f"Filtered df_raw dimensions: {df_raw.shape}")

df = df_raw.drop(columns=columns_to_drop)

# Display the cleaned DataFrame's dimensions and columns
print(f"Cleaned DataFrame dimensions: {df.shape}")
print("Columns after cleaning:", df.columns)

"""
CREATE PATIENT DESCRIPTION
"""
# print the cleaned dataframe
print("Cleaned DataFrame:")
# print the size of the dataframe
print(df.shape)
print(df)
"""
human friendly columns
"""

# in 'sex' column of the replace f with Femminile, m with Maschile
# Replace 'f' with 'Femminile' and 'm' with 'Maschile' in the 'sex' column
if "sex" in df.columns:
    df["sex"] = df["sex"].replace({"f": "Femminile", "m": "Maschile"})


# df = fill_missing_values(df)

if "aao" in df.columns:
    # Round the age to the nearest integer
    df["aao"] = df["aao"].round()


# bmi
def bmi_category(bmi_value):

    if bmi_value < 18.5:
        return "Sottopeso"
    elif 18.5 <= bmi_value <= 24.9:
        return "Normale"
    elif 25 <= bmi_value <= 29.9:
        return "Sovrappeso"
    elif bmi_value >= 30:
        return "Obeso"


if "bmi" in df.columns:
    df["bmi"] = df["bmi"].apply(bmi_category)

# see the df_vars content_human columns and for the variables that has si/no,  replace 1 with Si and 0 with No

# Step 1: Identify columns with 'si/no' in the content_human column
si_no_columns = df_vars[df_vars["content_human"] == "si/no"]["variable"].tolist()

# Step 2: Replace 1 with "Si" and 0 with "No" in those columns
for column in si_no_columns:
    if column in df.columns:
        df[column] = df[column].replace({1: "Si", 0: "No"})


# Step 1: Identify columns with 'si/no' in the content_human column
noe_no_columns = df_vars[df_vars["content_human"] == "normale/non normale"][
    "variable"
].tolist()

# Step 2: Replace 1 with "Si" and 0 with "No" in those columns
for column in noe_no_columns:
    if column in df.columns:
        df[column] = df[column].replace({0: "Normale", 1: "Non Normale"})

# Step 1: Identify columns with 'si/no' in the content_human column
el_no_columns = df_vars[df_vars["content_human"] == "elevata/nella norma"][
    "variable"
].tolist()

# Step 2: Replace 1 with "Si" and 0 with "No" in those columns
for column in el_no_columns:
    if column in df.columns:
        df[column] = df[column].replace({1: "Elevata", 0: "Nella Norma"})


# Display the updated DataFrame
print(df)

"""
clinical course mapping
"""
# Mapping for the variable 'cc' with Italian descriptions
cc_mapping = {
    "HP:0003685": "Stabile",  # stable
    "HP:0003676": "Progressivo",  # progressive
    "HP:0003682": "Intermittente",  # intermittent
    "HP:0025254": "Migliorato",  # improved by
    "HP:0025285": "Aggravato",  # aggravated by
}

if "cc" in df.columns:
    # Apply the mapping to the 'cc' column
    df["cc"] = df["cc"].map(cc_mapping)
    print("Clinical course mapping applied to df")
    print(df["cc"])


"""
add abnormality columns
"""

# remove old abnormalities columns
abnormality_columns = [
    "abbi",
    "abnb",
    "abncvs",
    "abndig",
    "abnear",
    "abneye",
    "abngd",
    "abngrow",
    "abngus",
    "abnhead",
    "abnint",
    "abnmss",
    "abnns",
    "abnoth",
    "abnresp",
    "abnskin",
]
# check if the columns are in the df and in case remove them
if set(abnormality_columns).issubset(df.columns):
    df = df.drop(abnormality_columns, axis=1)


mapping_abnormalities_path = os.path.join(
    saved_result_path, "mapping/mapping_abnormalities_HPO.xlsx"
)

mapping_symptoms_path = os.path.join(
    saved_result_path, "mapping/mapping_sympthoms.xlsx"
)

# Remove IT patients
print(df.columns)
# add abnormalities from HPO
df = add_abnormalities_cols(df, mapping_abnormalities_path, mapping_symptoms_path)
# check the cols of the df_abn and mantain only the abn that are in the selected features
df = df[selected_features]

# print dimensions and columns of the df
print(f"Cleaned DataFrame dimensions after affing abnormalities: {df.shape}")


# Define the column mapping
column_mapping = {
    "Anomalia Comportamentale/psichiatrica": "beh_psy_abn",
    "Anomalie Cardiache": "card_abn",
    "Anomalia Sistema Digestivo": "diges_abn",
    "Anomalie all'Orecchio e alla Voce": "ear_voice_abn",
    "Anomalie Oculari": "eye_abn",
    "Disturbo dell'andatura": "gait_abn",
    "Anomalia dello sviluppo prenatale o della nascita/crescita": "natal_growth_abn",
    "Anomalia Sistema genito-urinario o seno": "genit_breast_abn",
    "Anomalia Testa o collo": "head_neck_abn",
    "Anomalie di organi interni, endocrine o del sangue": "internal_abn",
    "Anomalie Muscolo-Scheletriche e Cutanee": "musc_sk_abn",
    "Anomalia Sistema nervoso": "nerv_abn",
    "Altra anomalia": "other_abn",
    "Anomalia Sistema respiratorio": "resp_abn",
    "Anomalia Tessuto connettivo / pelle": "conn_skin_abn",
}
# Reverse the mapping dictionary to map from abbreviated names to full descriptions
reversed_mapping = {v: k for k, v in column_mapping.items()}

# Rename the columns of the DataFrame when the column name is in the mapping dictionary
if set(column_mapping.keys()).issubset(df.columns):
    df = df.rename(columns=column_mapping)
    print("Abnormalities added to df")




df = df.rename(columns=reversed_mapping)

print("Abnormalities added to df")


"""
 sobstitute HPO code with the name of the symptom
"""

symptoms_mapping_path = os.path.join(
    saved_result_path, "mapping", "psterm_modify_association_Besta.xlsx"
)
# Identify columns containing 'symp_on' in their names
columns_with_symptoms = [col for col in df.columns if "symp_on" in col]
# print("Columns with symptoms:", columns_with_symptoms)

# Load symptoms mapping file
mapping_symptoms = pd.read_excel(symptoms_mapping_path)

# Clean the 'psterm__decod' column by removing 'HP:'
mapping_symptoms["psterm__decod"] = mapping_symptoms["psterm__decod"].str.replace(
    "HP:", ""
)

# Define a function to extract text between single quotes
def extract_text(text):
    match = re.search(r"'([^']*)'", text)
    return match.group(1) if match else None


#print the association_psterm__modify column
#print("Mapping symptoms:\n", mapping_symptoms)
# Apply the function to the 'associated_psterm__modify' column
mapping_symptoms["associated_psterm__modify"] = mapping_symptoms[
    "associated_psterm__modify"
].apply(extract_text)
mapping_symptoms["psterm__decod"] = mapping_symptoms["psterm__decod"].apply(
    lambda x: float(str(x).lstrip("0")) if str(x).lstrip("0") else 0.0
)

print("Mapping symptoms:\n", mapping_symptoms)

# Reset index of df_test to ensure unique index values
df.reset_index(drop=True, inplace=True)


# Substitute the symptom codes with names in df_test using the mapping file
symptoms_dict = mapping_symptoms.set_index("psterm__decod")[
    "associated_psterm__modify"
].to_dict()
for col in columns_with_symptoms:
    df[col] = df[col].map(symptoms_dict)

#print the  values of the columns with symptoms
for col in columns_with_symptoms:
    print("unique values in the column", col, df[col].value_counts())




# substitute the symptom names with the italian translation
mapping_en_ita_hpo_path = os.path.join(
    saved_result_path, "mapping", "mapping_en_ita_hpo.xlsx"
)


# substitute in the df the english symptom names (column EN) with the italian translation (column IT)
mapping_en_ita_hpo = pd.read_excel(mapping_en_ita_hpo_path)

mapping_en_ita_hpo = mapping_en_ita_hpo.drop_duplicates(subset="EN")

for col in columns_with_symptoms:
    df[col] = df[col].map(mapping_en_ita_hpo.set_index("EN")["IT"])

# check that each symptom has been translated
for col in columns_with_symptoms:
    print("unique values in the column", col, df[col].unique())


# Display the modified DataFrame to verify the changes
# print("Modified df_test with symptom names:\n", df_test[columns_with_symptoms])
print("sobsituted HPO code with the name of the symptom in df")




"""
create a unique list of symptoms
"""
# Step 1: Identify columns that contain 'symp_on'
symptom_columns = [col for col in df.columns if "symp_on" in col]

#print the symptom columns content
print("Symptom columns content:\n", df[symptom_columns])

# Step 2: Create a new column with consolidated symptoms
df["Sintomi all Insorgenza"] = df[symptom_columns].apply(
    lambda row: ", ".join(row.dropna().astype(str)), axis=1
)

#print the new column
print(df["Sintomi all Insorgenza"])

# Step 3: Remove the original symptom columns
df = df.drop(columns=symptom_columns)
# print that the new column was created
print("New column 'all_symptoms' created in df")





# Definizione della nuova mappatura per pssev
pssev_mapping = {
    "HP:0012827": "Limite",
    "HP:0012825": "Lieve",
    "HP:0012826": "Moderato",
    "HP:0012829": "Profondo",
    "HP:0012828": "Grave",
}


# print("columns to be renamed", df_test.columns)
# Aggiornamento della colonna pssev utilizzando la nuova mappatura
for index, row in df.iterrows():
    if row["pssev"] in pssev_mapping:
        df.at[index, "pssev"] = pssev_mapping[row["pssev"]]
# print("pssev updated", df_test["pssev"])

print("pssev mapping applied to df")


# rename pimgres column


rename_mapping = {
    1: "Cambiamenti specifici",
    2: "Cambiamenti aspecifici",
    3: "Nessuna progressione rispetto all'ultima imaging",
    0: "Normale",
}


# apply the mapping to the column pimgres
if "pimgres" in df.columns:
    df["pimgres"] = df["pimgres"].map(rename_mapping)
    # print("pimgres updated", df_test["pimgres"])
    print("pimgres mapping applied to df")




"""
make the age an int without decimal
"""


# when is not nan, convert the age to int
if "aao" in df.columns:
    # Converte la colonna aao in int senza decimali
    df["aao"] = df["aao"].apply(lambda x: int(x) if pd.notna(x) else x).astype("Int64")

    # df["aao"] = df["aao"].apply(lambda x: int(x) if pd.notna(x) else x)

print("age column in df is now int without decimal")


"""
rename all the columns of the df
"""
# rename the columns of the df_``
# rename the columns (their name is in the variable) with the name in the column name_human of the df_vars
rename_mapping = dict(zip(df_vars["variable"], df_vars["name_human"]))

# Renaming the columns of the df DataFrame
df = df.rename(columns=rename_mapping)

# order the columns of the df_test based on the order of the column "Order" of the df_vars

# Create a mapping dictionary for column order
order_mapping = df_vars.set_index("name_human")["Order"].to_dict()

# Split df_test columns into those with an order and those without
columns_with_order = [col for col in df.columns if col in order_mapping]
columns_without_order = [col for col in df.columns if col not in order_mapping]

# Sort the columns with a defined order
columns_with_order.sort(key=lambda x: order_mapping[x])

# Combine ordered and unordered columns
ordered_columns = columns_with_order + columns_without_order

# Reorder the columns of df_test
df = df[ordered_columns]


print("Columns renamed in df_test:\n", df.columns)
print(df)

#substitute 0 with No in the df
df = df.astype(object).replace(0, "No")

print("0 substituted with No in df")

# save in csv the resultin df_test
df.to_csv(
    os.path.join(SURVEY_PATH, "df_test_human_friendly_best.csv"),
    index=False,
)

'''
df.to_csv(
    os.path.join(SURVEY_PATH, "df_test_human_friendly_best_140.csv"), #to create the survey from raw df
    index=False,
)'
'''


# print that the df_test was saved
print("df_test saved in csv as df_test_human_friendly_best.csv in the survey folder")
#print the SURVEY_PATH
print(SURVEY_PATH)

#read the df_test_human_friendly_best_140 and df_test_human_friendly_best
df_test_human_friendly_best_140 = pd.read_csv(os.path.join(SURVEY_PATH, "df_test_human_friendly_best_140.csv"))
df_test_human_friendly_best = pd.read_csv(os.path.join(SURVEY_PATH, "df_test_human_friendly_best.csv"))

#find the subjid that are in df_test_human_friendly_best_140 but not in df_test_human_friendly_best
subjid_not_in_df_test_human_friendly_best = df_test_human_friendly_best_140[~df_test_human_friendly_best_140["id paziente"].isin(df_test_human_friendly_best["id paziente"])]
print("subjid that are in df_test_human_friendly_best_140 but not in df_test_human_friendly_best")
#for these partients, substitute No with -998
subjid_not_in_df_test_human_friendly_best = subjid_not_in_df_test_human_friendly_best.fillna(-998)
#add these subjid to the df_test_human_friendly_best
df_test_human_friendly_best = pd.concat([df_test_human_friendly_best, subjid_not_in_df_test_human_friendly_best])
#print the new df_test_human_friendly_best


#correct sex column
if "Sesso" in df_test_human_friendly_best.columns:
    df_test_human_friendly_best["Sesso"] = df_test_human_friendly_best["Sesso"].replace({"1": "Femminile", "0": "Maschile"})

print(df_test_human_friendly_best)
#save the new df_test_human_friendly_best
df_test_human_friendly_best.to_csv(
    os.path.join(SURVEY_PATH, "df_test_human_friendly_best_complete140.csv"),
    index=False,
)