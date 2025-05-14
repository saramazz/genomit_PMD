
# Standard Library Imports
import os
import sys
import time
import json
from collections import Counter
from itertools import combinations
from datetime import datetime
import os
import re

from sklearn.utils.class_weight import compute_class_weight

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
#Distribution patients for hospital
#Load 
# ask if consider patients with no sympthoms
Input = input("Do you want to consider the reduced df? (y/n)")  # Complete

if Input == "y":
    # Constants and Paths
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_no_symp.csv")  # Reduced
    # GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_red"
    )
else:
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_symp.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_compl"
    )


EXPERIMENT_PATH_RESULTS = os.path.join(EXPERIMENT_PATH, "results")

df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH, EXPERIMENT_PATH)

print (df.columns)

print("Number of patients in the df: ", len(df))
print(df.head(5))    

print("Loading and previewing global DataFrame...")
pickle_file_path = os.path.join(saved_result_path, "df/df_Global_raw.pkl")
df_row = pd.read_pickle(pickle_file_path)
#print the columns of the df_row
print(df_row.columns)

#print the number of patients in the df_row and for each hospital
print("Number of patients in the df_row: ", len(df_row))
hospital_counts_row = df_row["Hospital"].value_counts()
print("Number of patients in each hospital in df_row:")
print(hospital_counts_row)


#check if the sum of the patients in each hospital is equal to the number of patients in the df_row
hospital_counts_row_sum = hospital_counts_row.sum()
if hospital_counts_row_sum == len(df_row):
    print("The sum of the patients in each hospital is equal to the number of patients in the df_row")
else:
    print("The sum of the patients in each hospital is not equal to the number of patients in the df_row")
    print("The sum of the patients in each hospital is: ", hospital_counts_row_sum)
    print("The number of patients in the df_row is: ", len(df_row))
#check if the patients in df_row are in df

#add to df the column Hospital from df_row matching the subjid
df["Hospital"] = df_row.set_index("subjid")["Hospital"].reindex(df["subjid"]).values

#print the number of patients in each hospital
hospital_counts = df["Hospital"].value_counts()
print("Number of patients in the df: ", len(df))
print("Number of patients in each hospital:")
print(hospital_counts)

#check if the sum of the patients in each hospital is equal to the number of patients in the df
hospital_counts_sum = hospital_counts.sum()
if hospital_counts_sum == len(df):
    print("The sum of the patients in each hospital is equal to the number of patients in the df")
else:
    print("The sum of the patients in each hospital is not equal to the number of patients in the df")
    print("The sum of the patients in each hospital is: ", hospital_counts_sum)
    print("The number of patients in the df is: ", len(df))



#sex distribution of df
# Calculate the sex distribution in df
sex_counts = df['sex'].value_counts()
print("Sex distribution in df:")
print(sex_counts)
#print the percentage of

# Define labels for the pie chart
sex_labels = {0: 'Female', 1: 'Male'}
labeled_sex_counts = sex_counts.rename(index=sex_labels)

# Calculate and print the percentage of sex distribution
sex_percentage = (sex_counts / len(df)) * 100
print("Percentage of sex distribution:")
print(sex_percentage)


# Plot a pie chart of the sex distribution
plt.figure(figsize=(8, 8))
plt.pie(labeled_sex_counts, startangle=90)
#plt.title("Sex Distribution in df")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart to the experimental path
chart_path = os.path.join(EXPERIMENT_PATH, "sex_distribution_pie_chart.png")
plt.savefig(chart_path)
print(f"Pie chart saved at: {chart_path}")

# CAGE distribution
#add the column cage from df_row to df matching the subjid
df['cage'] = df_row.set_index("subjid")["cage"].reindex(df["subjid"]).values
cage_threshold_counts = df['cage'].apply(lambda x: '>=16' if x >= 16 else '<16').value_counts()
cage_threshold_percentage = (cage_threshold_counts / len(df)) * 100
print("\nCAGE distribution (threshold 16 years):")
print(cage_threshold_counts)
print("Percentage of CAGE distribution:")
print(cage_threshold_percentage)

plt.figure(figsize=(8, 8))
plt.pie(cage_threshold_counts, startangle=90, textprops={'color':"w"})
plt.title("CAGE Distribution (Threshold 16 years)")
plt.axis('equal')
chart_path = os.path.join(EXPERIMENT_PATH, "cage_distribution_pie_chart.png")
plt.savefig(chart_path)
print(f"CAGE pie chart saved at: {chart_path}")

# AAO distribution
aao_threshold_counts = df['aao'].apply(lambda x: '>=16' if x >= 16 else '<16').value_counts()
aao_threshold_percentage = (aao_threshold_counts / len(df)) * 100
print("\nAAO distribution (threshold 16 years):")
print(aao_threshold_counts)
print("Percentage of AAO distribution:")
print(aao_threshold_percentage)

plt.figure(figsize=(8, 8))
plt.pie(aao_threshold_counts, startangle=90, textprops={'color':"w"})
plt.title("AAO Distribution (Threshold 16 years)")
plt.axis('equal')
chart_path = os.path.join(EXPERIMENT_PATH, "aao_distribution_pie_chart.png")
plt.savefig(chart_path)
print(f"AAO pie chart saved at: {chart_path}")

#distribution Clionical Diagnosis

import matplotlib.pyplot as plt
# Mapping of clindiag_decode values to labels
clindiag_mapping = {
    'C01': 'MELAS', 'B01': 'CPEO', 'A02': 'ADOA', 'A01': 'LHON', 'C04': 'Leigh syndrome',
    'C19': 'Encephalopathy', 'B02': 'CPEO plus', 'C03': 'MERRF', 'B03': 'MiMy (without PEO)',
    'E': 'unspecified mitochondrial disorder', 'C06': 'Kearns-Sayre-Syndrome (KSS)', 'C05': 'NARP',
    'C18': 'Encephalomyopathy', 'C02': 'MIDD', 'C17': 'Other mitochondrial multisystem disorder',
    'C07': 'SANDO/MIRAS/SCAE', 'F': 'asymptomatic mutation carrier', 'D01': 'Isolated mitochondrial Cardiomyopathy',
    'A03': 'other MON', 'C08': 'MNGIE', 'C16': 'LBSL', 'C': 'Mitochondrial Multisystem Disorders',
    'C09': 'Pearson syndrome', 'C12': 'Wolfram-Syndrome (DIDMOAD-Syndrome)',
    'D05': 'Other mitochondrial mono-organ disorder'
}
# Replace 'clindiag__decod' values with their corresponding text descriptions
#add clindiag__decod to df from df_row based on subjid
df['clindiag__decod'] = df_row.set_index("subjid")["clindiag__decod"].reindex(df["subjid"]).values


df['clindiag__text'] = df['clindiag__decod'].map(clindiag_mapping)

# Create a histogram using the text values

plt.figure(figsize=(20, 10))

df['clindiag__text'].value_counts().plot(kind='bar', color='skyblue')

#plt.title('Histogram of clindiag__decod')

plt.xlabel('Clinical Diagnosis')

plt.ylabel('Frequency')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()

#save the figure
plt.savefig(os.path.join(EXPERIMENT_PATH, "clindiag_histogram.png"), dpi=300, bbox_inches='tight')

#Distribution 1,2,3 sympthom onset

"""
sobstitute HPO code with the name of the symptom
"""

#reduce the df to only test subjid using the column test
df = df[df["test"] == 1]
symptoms_mapping_path = os.path.join(
        saved_result_path, "mapping", "psterm_modify_association_Besta.xlsx"
    )
# Identify columns containing 'symp_on' in their names
columns_with_symptoms = [col for col in df.columns if "symp_on" in col]
print("Columns with symptoms:", columns_with_symptoms)

# Load the mapping from the Excel file
mapping_symptoms = pd.read_excel(symptoms_mapping_path)

# Ensure 'psterm__decod' codes are treated as strings
mapping_symptoms['psterm__decod'] = mapping_symptoms['psterm__decod'].astype(str)

# Define a function to clean and format the codes, handling NaN and 'nan' strings
def clean_code(code):
    try:
        # Remove 'HP:' if present, and leading zeros
        cleaned_code = str(code).replace('HP:', '').lstrip('0')
        return cleaned_code
    except (ValueError, TypeError):
        return None

# Create a dictionary for mapping codes to symptom names, cleaning the keys
symptoms_dict = {
    clean_code(key): value 
    for key, value in mapping_symptoms.set_index('psterm__decod')['associated_psterm__modify'].to_dict().items()
    if clean_code(key) is not None
}

# Identify columns containing 'symp_on' in their names
columns_with_symptoms = [col for col in df.columns if "symp_on" in col]

# Convert columns to strings and map them to names using the cleaned dictionary
for col in columns_with_symptoms:
    df[col] = df[col].apply(lambda x: symptoms_dict.get(clean_code(x), x))

# Define a function to clean the bracketed entries
def clean_symptom_name(symptom):
    if pd.isna(symptom):
        return symptom
    return str(symptom).strip("[]' ")

# Apply the function to each column with symptoms
for col in columns_with_symptoms:
    df[col] = df[col].apply(clean_symptom_name)

# Display the modified DataFrame to verify the changes
print("Modified df with symptom names:\n", df[columns_with_symptoms].head(5))
print("Substituted HPO code with the name of the symptom in df.")


# extract columns names that contain Symptom
symptoms_columns = columns_with_symptoms
print(symptoms_columns)
# Assicurati che i dati nella colonna dei sintomi siano trattati come stringhe
for symptoms_column in df.columns:
    if "symp" in symptoms_column:
        print("Symptoms column:", symptoms_column)
        # Rimuovi eventuali valori mancanti o non validi
        df = df.dropna(subset=[symptoms_column])
        #remove -998
        df = df[df[symptoms_column] != -998]
        df = df[df[symptoms_column] != "-998"]

        #remove numeric values
        df = df[~df[symptoms_column].astype(str).str.isnumeric()]
        df[symptoms_column] = df[symptoms_column].astype(str)

        # Calcola le frequenze delle parole
        word_counts = Counter(df[symptoms_column])
        # Rimuovi 'nan' dalla conta
        word_counts.pop("nan", None)

        # Parametri per il word cloud
        max_font_size = 80  # Dimensione massima del font per le parole più frequenti
        colormap = plt.cm.get_cmap("magma")  # Colormap più scura e intensa (magma)       

        # Convert Counter object to a pandas DataFrame for easier plotting
        word_counts_df = pd.DataFrame.from_dict(
            word_counts, orient="index", columns=["Frequency"]
        )
        print(word_counts_df)

        # Sort words by frequency
        word_counts_df = word_counts_df.sort_values(by="Frequency", ascending=False)
        # remove if frequency is less than 1
        word_counts_df = word_counts_df[word_counts_df["Frequency"] > 2]

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(word_counts_df.index, word_counts_df["Frequency"], color="lightblue")
        plt.xlabel("Symptom")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right", fontsize=13)
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            os.path.join(
                EXPERIMENT_PATH,f"histogram_{symptoms_column}.png"
            ),
            format="png",
        )
        plt.close()
        print(f"Histogram saved for {symptoms_column} at: {os.path.join(EXPERIMENT_PATH,'histogram_{symptoms_column}.png')}")


