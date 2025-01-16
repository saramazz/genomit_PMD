"""
Code to analyze the patients in the training and test set and the patients excluded from the training and test set.
Histograms of the missing values in the full dataset, in the training and test set, and in the excluded patients are created.
"""

# Local imports
from config import global_path, saved_result_path_classification
from utilities import *
from preprocessing import *
from processing import *
from plotting import *


# Standard library imports
import os
import sys
import time
from datetime import datetime
from collections import Counter
from itertools import combinations
import json

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
import shap


# Analyis patients gendna and variables missing in the db and in htem

# Redirect the standard output to a file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
# file_name = f"classification_reports_FINAL.txt"

# Redirect the standard output to the file
# sys.stdout = open(os.path.join(saved_result_path_classification, file_name), "w")


"""
# load the full data
"""
# Load Global DataFrame with only first visit data
hospital_name = "Global"
df_path = os.path.join(saved_result_path, "df", "df_preprocessed_Global.pkl")

df = pd.read_pickle(df_path)


# Display the dimensions and columns of the DataFrame
nRow, nCol = df.shape
print(
    f'The DataFrame "df_preprocessed" from {hospital_name} hospital contains {nRow} rows and {nCol} columns.'
)


# load the train and test data
# Define experiment parameters and split the data saving in a pickle file IMPORTING FROM THE PREVIOUS SCRIPT
config_path = os.path.join(
    saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)
config_dict = pd.read_pickle(config_path)

test_subjects = config_dict["test_subjects"]


# print test subjects
print("Test subjects:", test_subjects)
X_train = config_dict["X_train"]
X_test = config_dict["X_test"]

# print dimension of X_train and X_test
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Load the important variables from config
kf = config_dict["kf"]
scorer = config_dict["scorer"]
thr = config_dict["thr"]
nFeatures = config_dict["nFeatures"]
num_folds = config_dict["num_folds"]

# read the features from the file features_path = os.path.join(saved_result_path_classification, "features.txt")
features_path = os.path.join(saved_result_path_classification, "features.txt")
with open(features_path, "r") as file:
    features = [line.strip() for line in file]

# remove gendna_type from features
if "gendna_type" in features:
    features.remove("gendna_type")

# print the features
print("Features names:", features)


# find the patients excluded from the train and test data
# Save patients with NaN 'gendna' information to a file
excluded_patients_path = os.path.join(
    saved_result_path_classification, "patients_with_nan_or_1_gendna.csv"
)
# Load the excluded patients
excluded_patients = pd.read_csv(excluded_patients_path)
# Display the dimensions and columns of the DataFrame
nRow, nCol = excluded_patients.shape
print(f'The DataFrame "excluded_patients" contains {nRow} rows and {nCol} columns.')

# mantain only the column features in the excluded_patients and in the df
df = df[features]
excluded_patients = excluded_patients[features]


# print the new dimensions
print("Dimension df ", df.shape)
print("Dimension excluded: ", excluded_patients.shape)

# compare distribution of all the variable in all dataset, in the training and in the excluded patients
# Calculate the percentage of missing values
missing_df = df.isnull().mean() * 100
missing_excluded_patients = excluded_patients.isnull().mean() * 100

# Sort the percentages in descending order
missing_df_sorted = missing_df.sort_values(ascending=False)
missing_excluded_sorted = missing_excluded_patients.sort_values(ascending=False)

# Print dimensions
print("Dimensions of df:", df.shape)
print("Dimensions of excluded_patients:", excluded_patients.shape)


# Function to create a bar plot
def plot_missing_data(missing_data, title):
    plt.figure(figsize=(28, 15))
    missing_data.plot(kind="bar")
    plt.title(f"Missing Values Percentage in {title}")
    plt.xlabel("Features")
    plt.ylabel("Percentage of Missing Values")
    plt.xticks(rotation=45)
    # save it
    my_file = "Histogram_MissingValues_" + title + ".png"
    plt.savefig(
        os.path.join(saved_result_path_classification, my_file), bbox_inches="tight"
    )
    plt.show()


# Create bar plots
plot_missing_data(missing_df_sorted, "df")
plot_missing_data(missing_excluded_sorted, "excluded_patients")

# print that the plots have being saved to the folder
print("Plots saved in the folder:", saved_result_path_classification)


# Calculate missing values percentage
missing_df = df.isnull().mean() * 100
missing_excluded_patients = excluded_patients.isnull().mean() * 100

# Calculate the difference in missing values
missing_difference = missing_excluded_patients - missing_df

# Create a DataFrame for better visualization
missing_comparison = pd.DataFrame(
    {
        "df_missing_percent": missing_df,
        "excluded_patients_missing_percent": missing_excluded_patients,
        "difference_percent": missing_difference,
    }
).sort_values(by="excluded_patients_missing_percent", ascending=False)

# Display the comparison
print("Comparison of missing values percentages:\n")
print(missing_comparison)


# Create a grouped bar plot for missing values
def plot_missing_comparison(missing_comparison, save_path):
    # Sort the DataFrame by "difference_percent" in descending order
    missing_comparison = missing_comparison.sort_values(
        by="df_missing_percent", ascending=False
    )

    # Define the positions for each bar group
    x = np.arange(len(missing_comparison.index))

    # Set bar width
    bar_width = 0.25

    # Create the grouped bar plot
    plt.figure(figsize=(20, 10))
    plt.bar(
        x - bar_width,
        missing_comparison["df_missing_percent"],
        width=bar_width,
        label="Total dataset",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x,
        missing_comparison["excluded_patients_missing_percent"],
        width=bar_width,
        label="Excluded patients",
        color="orange",
        alpha=0.7,
    )
    plt.bar(
        x + bar_width,
        missing_comparison["difference_percent"],
        width=bar_width,
        label="Difference tot-excl",
        color="green",
        alpha=0.7,
    )

    # Add labels and title
    plt.title("Comparison of Missing Values Percentages by Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Percentage of Missing Values", fontsize=14)
    plt.xticks(x, missing_comparison.index, rotation=45, ha="right", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plot_file = "GroupedBar_MissingValues_Comparison.png"
    plt.savefig(os.path.join(save_path, plot_file), bbox_inches="tight")
    plt.show()

    # Confirm that the plot has been saved
    print(f"Grouped bar plot saved in the folder: {save_path}")


# print an hist of the difference_percent in the missing values, df_missing_percent  excluded_patients_missing_percent in a single hist
# Plot the grouped bar chart
plot_missing_comparison(missing_comparison, saved_result_path_classification)

#print the df_missing_percent
#sort the missing_comparison by df_missing_percent
missing_comparison = missing_comparison.sort_values(by="df_missing_percent", ascending=False)
print(missing_comparison["df_missing_percent"])
#save missing_comparison["df_missing_percent"] to a file
missing_comparison["df_missing_percent"].to_csv(os.path.join(saved_result_path_classification, "missing_comparison_df_missing_percent.csv"))

# print the distribution of missing values in the full data and in the train+ test set and in the patient excluded
# Create a grouped bar plot for missing values
def plot_missing_tot_df(missing_comparison, save_path):
    # Sort the DataFrame by "difference_percent" in descending order
    missing_comparison = missing_comparison.sort_values(
        by="df_missing_percent", ascending=False
    )

    # Define the positions for each bar group
    x = np.arange(len(missing_comparison.index))

    # Set bar width
    bar_width = 0.25

    # Create the grouped bar plot
    plt.figure(figsize=(20, 10))
    plt.bar(
        x - bar_width,
        missing_comparison["df_missing_percent"],
        width=bar_width,
        label="Total dataset",
        color="blue",
        alpha=0.7,
    )

    # Add labels and title
    plt.title("Comparison of Missing Values Percentages by Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Percentage of Missing Values", fontsize=14)
    plt.xticks(x, missing_comparison.index, rotation=45, ha="right", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plot_file = "GroupedBar_MissingValues_Tot_df.png"
    plt.savefig(os.path.join(save_path, plot_file), bbox_inches="tight")
    plt.show()

    # Confirm that the plot has been saved
    print(f"Grouped bar plot saved in the folder: {save_path}")


# print an hist of the difference_percent in the missing values, df_missing_percent  excluded_patients_missing_percent in a single hist
# Plot the grouped bar chart
plot_missing_tot_df(missing_comparison, saved_result_path_classification)
