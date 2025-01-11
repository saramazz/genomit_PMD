"""
BEST MODEL: RF
code to do classification of nDNA vs mtDNA experimenting features, balancement and classifiers
RandomForestClassifier+SMOTE

Best params: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 250}
Best estimator: RandomForestClassifier(max_depth=10, n_estimators=250)
Best score: 0.8020089142834681
accuracy                           0.74       176
   macro avg       0.73      0.72      0.72       176
weighted avg       0.74      0.74      0.74       176

#use the train index instead of train_test_split

"""

# Standard library imports
import os
import sys
import time
from collections import Counter
from itertools import combinations
import json


### Third-party library imports:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    LeaveOneGroupOut,
    KFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    mean_squared_error,
    f1_score,
    make_scorer,
)
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier

# from catboost import CatBoostClassifier
from scipy.stats import mode
import shap
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier

# import for datetime
from datetime import datetime


### Locally defined modules:

from config import global_path, saved_result_path_classification
from utilities import *
from processing import *
from preprocessing import *
from plotting import *

# Set the random seed for reproducibility
np.random.seed(42)

"""
IMPORT DATA
"""

# Redirect the standard output to a file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"classification_reports.txt"  # _{current_datetime}.txt"

# Redirect the standard output to the file
# sys.stdout = open(os.path.join(saved_result_path_classification, file_name), "w")


# Load Global DataFrame with only first visit data
hospital_name = "Global"
df_path = os.path.join(saved_result_path, "df", "df_preprocessed_Global.pkl")

df = pd.read_pickle(df_path)


# Display the dimensions and columns of the DataFrame
nRow, nCol = df.shape
print(
    f'The DataFrame "df_preprocessed" from {hospital_name} hospital contains {nRow} rows and {nCol} columns.'
)


# print data info

# Filter out NaN values for each column
df_sex = df["sex"].dropna()
df_aao = df["aao"].dropna()
df_cage = df["cage"].dropna()


# Calculate percentages for aao below and above 16 years old
aao_below_16 = df_aao[df_aao < 16]
aao_below_16_percentage = (len(aao_below_16) / len(df_aao)) * 100
aao_above_16 = df_aao[df_aao >= 16]
aao_above_16_percentage = (len(aao_above_16) / len(df_aao)) * 100

print(
    f"Percentage of subjects with Age at Onset (aao) below 16 years old: {aao_below_16_percentage:.2f}%"
)
print(
    f"Percentage of subjects with Age at Onset (aao) above 16 years old: {aao_above_16_percentage:.2f}%"
)

# Calculate percentages for cage below and above 16 years old
cage_below_16 = df_cage[df_cage < 16]
cage_below_16_percentage = (len(cage_below_16) / len(df_cage)) * 100
cage_above_16 = df_cage[df_cage >= 16]
cage_above_16_percentage = (len(cage_above_16) / len(df_cage)) * 100

print(
    f"Percentage of subjects with Calculated Age (cage) below 16 years old: {cage_below_16_percentage:.2f}%"
)
print(
    f"Percentage of subjects with Calculated Age (cage) above 16 years old: {cage_above_16_percentage:.2f}%"
)

# Calculate sex distribution percentages
sex_counts = df_sex.value_counts()
total_count = len(df_sex)

percentage_male = (sex_counts["m"] / total_count) * 100
percentage_female = (sex_counts["f"] / total_count) * 100

print(f"Percentage male: {percentage_male:.2f}%")
print(f"Percentage female: {percentage_female:.2f}%")
print(f"Total percentage: {percentage_male + percentage_female:.2f}%")

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
df_clindiag = df
df_clindiag["clindiag__decod"] = df_clindiag["clindiag__decod"].map(clindiag_mapping)


# Create a histogram using the text values

plt.figure(figsize=(10, 6))

df_clindiag["clindiag__decod"].value_counts().plot(kind="bar", color="skyblue")

# plt.title("Histogram of clindiag__decod")

plt.xlabel("Clinical Diagnosis")

plt.ylabel("Number of patients")
# increase font size
plt.xticks(fontsize=12)

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    os.path.join(saved_result_path_classification, "clindiag_hist.png"),
    format="png",
    bbox_inches="tight",
)
plt.close()


# print('Columns:', df.columns)

# print distribution of classes, drop nans and 1, convert to numerical and print class distribution
df, df_not_numerical = process_gendna_column(df)

# insert Not applicable as -1
df = fill_missing_values(df)

# print(df)

"""
FEATURE SELECTION
"""

# Load the important variables from Excel
important_vars_path = os.path.join(global_path, "data", "important_variables.xlsx")
df_vars = pd.read_excel(important_vars_path)

# Specify the column name for considering variables
column_name = "consider for mtDNA vs nDNA classification?"

# Get the list of columns to drop based on 'N' in the specified column
columns_to_drop = list(df_vars.loc[df_vars[column_name] == "N", "variable"])

# print the columns of the df
# print("Columns of the df:", df.columns)
# Additional columns to drop
additional_columns_to_drop = [
    "Hospital",
    "nDNA",
    "mtDNA",
    # "gendna_type",
    "epiphen",
    "sll",
    "clindiag__decod",
    "encephalopathy",
]
additional_columns_to_drop += [col for col in df.columns if "pimgtype" in col]
additional_columns_to_drop += [col for col in df.columns if "psterm" in col]

columns_to_drop = columns_to_drop + additional_columns_to_drop

# Remove columns containing 'gendna'
columns_to_drop = [col for col in columns_to_drop if "gendna" not in col]
print("Columns to drop:", columns_to_drop)


# Sostituisci i valori mancanti con 998
df = df.fillna(998)
df_raw = df.copy()  # save the raw data non numerical

# Drop the columns from the DataFrame and convert to numerical
# X, y, X_df = define_X_y(df, columns_to_drop)

y = df["gendna_type"]

# Load the important variables from Excel
important_vars_path = os.path.join(global_path, "data", "important_variables.xlsx")
df_vars = pd.read_excel(important_vars_path)

# Specify the column name for considering variables
column_name = "consider for mtDNA vs nDNA classification?"


# Get the list of columns to drop based on 'N' in the specified column
df.drop(columns=columns_to_drop, inplace=True)

# Assign the processed dataframe to X_df
X_df = df

# Convert X_df to numpy array
X = X_df.values

# Print the type and dimension of X and y
print("\nType and Dimension of X:")
print(type(X), X.shape)

print("\nType and Dimension of y:")
print(type(y), y.shape)


"""
splitting into training and testing
"""


# import test_subjects from the previous experimentation script

# Define experiment parameters and split the data saving in a pickle file IMPORTING FROM THE PREVIOUS SCRIPT
config_path = os.path.join(
    saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)
config_dict = pd.read_pickle(config_path)

test_subjects = config_dict["test_subjects"]


# Creating the training and test sets
test_set = df[df["subjid"].isin(test_subjects)]
train_set = df[~df["subjid"].isin(test_subjects)]
X_train = train_set.drop(columns=["subjid", "gendna"])
X_test = test_set.drop(columns=["subjid", "gendna"])


features = X_train.columns

print("X_train shape:", X_train.shape)


# Load the important variables from config
kf = config_dict["kf"]
scorer = config_dict["scorer"]
thr = config_dict["thr"]
nFeatures = config_dict["nFeatures"]
num_folds = config_dict["num_folds"]


print("Features names:", features)

# convert features to a csv to save it
features = list(features)
features_path = os.path.join(saved_result_path_classification, "features.txt")

with open(features_path, 'w') as file:
    for item in features:
        file.write(f"{item}\n")

feature_names_df = pd.DataFrame(features, columns=["Feature Names"])
# save the features in a file

feature_names_df.to_csv(features_path, index=False, header=False)
# pause 40 sec
time.sleep(40)


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__


# ask if to continue
print("Do you want to continue with the classification?")
print("Press 'y' to continue or 'n' to stop")
answer = input()
if answer == "n":
    sys.exit()


"""
OVERSAMPLING AND CLASSIFICATION
"""

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Apply oversampling to training data
oversampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_resampled))


print("Starting the classification...")


classifiers = {
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
    )
}
"""
#select the best RF
classifiers = {
    "RandomForestClassifier": (
        RandomForestClassifier(),
        {
            "n_estimators": [100],  # Increased to 3 values
            "max_depth": [20],  # 4 values
            "min_samples_split": [5],  # 3 values
            "min_samples_leaf": [1],  # 3 values
            "max_features": ["sqrt"],  # 2 values
            "bootstrap": [False],  # 2 values
            "criterion": ["gini"],  # 2 values
        },
    )
}
"""

'''
#uncomment to fit the model

# Selecting the RandomForestClassifier
clf_name, (clf, param_grid) = list(classifiers.items())[0]


# Perform grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=kf,
    scoring=scorer,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
best_score_ = grid_search.best_score_

cv_results = grid_search.cv_results_
print("CV RESULTS:______________________________________________________")
print("Best params:", best_params)
print("Best estimator:", best_estimator)
print("Best score:", best_score_)


# Make predictions on the test set
y_pred = best_estimator.predict(X_test)

# Print the best parameters from grid search
print("Best Parameters:")
print(grid_search.best_params_)


"""'
Import if needed 
"""
'''
res_path = os.path.join(saved_result_path_classification, "classification_results.pkl")


with open(res_path, "rb") as f:
    results_dict = pickle.load(f)

# Extract variables
best_params = results_dict["best_params"]
best_estimator = results_dict["best_estimator"]
best_score_ = results_dict["best_score"]
cv_results = results_dict["cv_results"]
y_pred = results_dict["y_pred"]
y_test = results_dict["y_test"]
classification_report_dict = results_dict["classification_report"]
conf_matrix = results_dict["confusion_matrix"]
importances = results_dict["importances"]
feature_importance_data = results_dict["feature_importance_data"]

"""
Print the performances, confusion matrix, classification report, and feature importances
"""
print("CV RESULTS:______________________________________________________")
print("Best params:", best_params)
print("Best estimator:", best_estimator)
print("Best score:", best_score_)


# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
confusion_matrix_file = f"cm_best_model.png"
plot_confusion_matrix(
    y_test,
    y_pred,
    os.path.join(saved_result_path_classification, confusion_matrix_file),
)
plt.close()


print("Calculating and Plotting Importances...")

if hasattr(best_estimator, "feature_importances_"):

    importances = best_estimator.feature_importances_

    # print("Importances:", importances)
    # print("Features:", features)
    feature_importances = {features[i]: importances[i] for i in range(len(importances))}
    indices_all = np.argsort(importances)  # Sort indices by importance
    feature_importance_data = {
        "feature_importances": feature_importances,
        "top_10_features": {features[i]: importances[i] for i in indices_all},
    }
    # print("Feature Importances data:", feature_importance_data)

    # Plot ALL feature importances
    plt.figure(figsize=(10, 8))
    plt.title("All feature Importances", fontsize=15)
    plt.barh(
        range(len(indices_all)),
        importances[indices_all],
        color="lightblue",
        align="center",
    )
    # update feature names
    mapping_path = os.path.join(
        saved_result_path, "mapping", "mapping_variables_names.xlsx"
    )
    mapping_df = pd.read_excel(mapping_path)

    # look for the columns in the mapping file in the column variable and substitute it with the name in 'label' column
    variable_to_label = dict(zip(mapping_df["variable"], mapping_df["label"]))
    features = [variable_to_label[feature] for feature in features]
    plt.yticks(
        range(len(indices_all)),
        [features[i] for i in indices_all],
        ha="right",
        fontsize=10,
    )
    plt.xlabel("Relative Importance", fontsize=15)
    feature_importance_file = f"feature_imp_ALL_.png"
    plt.savefig(
        os.path.join(saved_result_path_classification, feature_importance_file),
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    # Plot ONLY top 10 feature importances
    indices = np.argsort(importances)[-10:]  # Select the top 10 most important features
    plt.title("Top 10 Feature Importances", fontsize=15)
    plt.barh(
        range(len(indices)), importances[indices], color="lightblue", align="center"
    )  # Use light blue color
    plt.yticks(
        range(len(indices)), [features[i] for i in indices], ha="right", fontsize=10
    )  # Rotate labels
    plt.xlabel("Relative Importance", fontsize=15)

    feature_importance_file = f"feature_imp_.png"
    plt.savefig(
        os.path.join(saved_result_path_classification, feature_importance_file),
        format="png",
        bbox_inches="tight",
    )
    plt.close()
else:
    print("No importances available")


# Save the best estimator to a file
best_estimator_file = f"best_estimator_.pkl"
with open(
    os.path.join(saved_result_path_classification, best_estimator_file), "wb"
) as f:
    pickle.dump(best_estimator, f)
print(f"Best estimator saved to {best_estimator_file}")

## Save all relevant results to a file
results_to_save = {
    "best_params": best_params,
    "best_estimator": best_estimator,
    "best_score": best_score_,
    "cv_results": cv_results,
    "y_pred": y_pred,
    "y_test": y_test,
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": conf_matrix,
    "importances": importances,
    "feature_importance_data": feature_importance_data,
}
"""
# Path to save the results
results_file_path = os.path.join(
    saved_result_path_classification, "classification_results.pkl"
)
with open(results_file_path, "wb") as f:
    pickle.dump(results_to_save, f)
    """

"""
MISCLASSIFICATION FN,FP ANALYSIS  
"""

# Identify false positives (FP) and false negatives (FN)
fp_indices = (y_test == 0) & (y_pred == 1)  # mt as nDNA their gen is 5,7
fn_indices = (y_test == 1) & (y_pred == 0)  # nDNA as mt, their gen is 4,6,8

# Get the subject IDs for FP and FN cases
fp_subjids = test_set.loc[fp_indices, "subjid"]
fn_subjids = test_set.loc[fn_indices, "subjid"]

# create a df with the test_set and the predictions and the fn and fp
test_set["predictions"] = y_pred
test_set["fp"] = fp_indices
test_set["fn"] = fn_indices

# print(test_set)

# print("False Positive (FP) Subject IDs:")
# print(fp_subjids.values)

# print("False Negative (FN) Subject IDs:")
# print(fn_subjids.values)

# add gendna column from df_not_numerical to the test_set basing on subjid
test_set["gendna_non_num"] = test_set["subjid"].map(
    df_not_numerical.set_index("subjid")["gendna"]
)

# print the test_set the gendna_type for the fp_subjids and fn_subjids
print("Class 0 mtDNA (5,7), Class 1 nDNA (4,6,8)")
print("False Positive (FP) class and gendna_non_num, in reality mt 5,7:")
print(test_set.loc[fp_indices, "gendna_type"].values)
# print(test_set.loc[fp_indices, "gendna"].values)
print(test_set.loc[fp_indices, "gendna_non_num"].values)
print("False Negative (FN) class and gendna_non_num, in reality n 4,6,8:")
print(test_set.loc[fn_indices, "gendna_type"].values)
# print(test_set.loc[fn_indices, "gendna"].values)
print(test_set.loc[fn_indices, "gendna_non_num"].values)

# save in excel the test_set
test_set.to_excel(os.path.join(saved_result_path_classification, f"test_set_best.xlsx"))
print("test_set numerical df saved in test_set_best.xlsx")

# create from the df_not_numerical the df with the subjid and the gendna
# Drop columns from df_not_numerical
# remove from columns_to_drop the columns_to_add

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

df_test = df_not_numerical[
    df_not_numerical["subjid"].isin(test_subjects)
]  # create df with not numerical data

# Drop columns from df_test
df_test.drop(columns=columns_to_drop, inplace=True)

print(
    "df_not_numerical shape after gendna processing, filling NAs, dropping columns of X and adding some for more explainability:",
    df_not_numerical.shape,
)


# add todf_test the prediction, fp, fn based on subjid
df_test["predictions"] = y_pred
df_test["fp"] = fp_indices
df_test["fn"] = fn_indices
# print shape of df_test
print("Added to df_test the FP, FN and y_pred :", df_test.shape)


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
df_test.to_excel(os.path.join(saved_result_path_classification, f"df_test_best.xlsx"))

print("df_test saved in df_test_best.xlsx")


"""
FEATURE IMPORTANCE

"""


# importances = results_dict["importances"]
feature_importance_data = results_dict["feature_importance_data"]
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
    sorted(renamed_feature_importances.items(), key=lambda item: item[1], reverse=True)
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
    data, columns=["Number of Features", "Last feature added", "Percentage Explained"]
)

print("data of the progress of the importance:")
# Print DataFrame
print(df)

# Save the df of the progress of the importance
df.to_excel(
    os.path.join(saved_result_path_classification, f"feature_importance_progress.xlsx")
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


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__
