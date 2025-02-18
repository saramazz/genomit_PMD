import os
import time

### Third-party Library Imports:
import numpy as np
import pandas as pd


### Local Imports:
from config import *
from utilities import *
from processing import *
from plotting import *

# import confusion matrix
from sklearn.metrics import confusion_matrix

# import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import shap


"""
Functions to plot for preprocessing and classification
"""

"""
PLOT PREPROCESSING
"""


def plot_missing_values(df, saved_result_path, file_name):

    # Columns to remove with in the name pimgtype and psterm__decod
    columns_to_remove = [
        col for col in df.columns if "pimgtype" in col or "psterm__decod" in col
    ]

    df = df.drop(columns=columns_to_remove)
    # Calculate missing values and their percentages
    missing_values = df.isnull().sum()
    total_values = len(df)
    missing_percentages = (missing_values / total_values) * 100
    missing_data = pd.DataFrame(
        {"Missing Values": missing_values, "Missing Percentage": missing_percentages}
    )
    # Sort the data by missing percentage
    missing_data = missing_data.sort_values(by="Missing Values", ascending=False)

    # Plotting
    plt.figure(figsize=(15, 25))
    plt.barh(missing_data.index, missing_data["Missing Percentage"], color="lightcoral")
    plt.xlabel("Percentage of Missing Values (%)")
    plt.ylabel("Variable Name")
    plt.title("Missing Values Percentage for Each Variable")
    plt.gca().invert_yaxis()
    # Decrease the font size on the y-axis
    plt.yticks(fontsize=10)
    plt.grid()
    plt.savefig(os.path.join(saved_result_path, file_name), bbox_inches="tight")

    # Print statement to confirm the plot has been saved
    print(f"Missing values plot saved in {saved_result_path}")
    plt.close()


def plot_phenotype_distribution(df, phenotype_column, saved_result_path):
    """
    Plot and save the distribution of phenotypes in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the phenotypes.
    - phenotype_column (str): The column name representing the phenotypes.
    - saved_result_path (str): The path to save the plot.

    Returns:
    - None
    """

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    df[phenotype_column].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Distribution of Phenotypes")
    plt.xlabel("Phenotype")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")  # Adjust rotation for better readability
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(saved_result_path, "phenotype_distribution.png")
    plt.savefig(save_path)
    plt.close()

    # Show the plot
    # plt.show()


def plot_histogram_visits_per_patient(df, hospital_name, saved_result_path):
    visit_count = (
        df.groupby("subjid")["visdat"].nunique().reset_index(name="num_visits")
    )
    unique_values = visit_count["num_visits"].unique()
    print("Unique values in 'num_visits':", unique_values)
    n, bins, patches = plt.hist(
        visit_count["num_visits"],
        bins=range(0, 10),
        align="left",
        color="#0504aa",
        alpha=0.7,
    )
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
    for i in range(len(patches)):
        plt.text(
            patches[i].get_x() + patches[i].get_width() / 2.0,
            patches[i].get_height(),
            str(int(n[i])),
            ha="center",
            va="bottom",
        )
    plt.xlabel("Number of Visits")
    plt.ylabel("Number of Patients")
    plt.title("Histogram of Number of Visits per Patient")
    plt.xlim(0, 8)
    my_file = "Histogram_VisitsxPatient_" + hospital_name + ".png"
    plt.savefig(os.path.join(saved_result_path, my_file), bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_gendna_distribution(df):
    """
    Plot the distribution of 'gendna_type_num' classes in a pie chart.

    Parameters:
    - df: DataFrame containing the 'gendna_type_num' column.
    - saved_result_path_classification: Path to save the generated plot.
    """

    plt.rcParams["font.size"] = 20

    # Validate necessary column existence
    if "gendna_type" not in df.columns:
        raise ValueError("DataFrame must contain 'gendna_type_num' column")

    # Calculate class counts directly from 'gendna_type_num'
    # class_counts = df["gendna_type_num"].value_counts()
    class_counts = df["gendna_type"].value_counts()

    # Set up the labels for the classes
    class_labels = {0: "mtDNA", 1: "nDNA"}
    class_counts.index = class_counts.index.map(class_labels.get)

    # Debug statements to check accuracy of class counts
    print(f"Class counts after mapping: {class_counts}")

    # Use seaborn color palette for shades of blue
    colors = sns.color_palette("Blues", len(class_counts))

    # Prepare and display the pie chart
    plt.figure(figsize=(15, 8))
    class_counts.plot.pie(
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        labels=class_counts.index,
        colors=colors,
        legend=True,
    )

    # Set title and other formatting elements
    plt.title("Distribution of nDNA vs mtDNA Classes", fontsize=20)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=20)

    # Save the plot to the designated path
    my_file = "gendna_distribution_pie_n_mt.png"
    plot_path = os.path.join(saved_result_path_classification, my_file)
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"gendna distribution plot saved at {plot_path}")
    plt.close()


def plot_clindiag_histogram(df_global):
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

    # Replace 'clindiag__decod' values with their corresponding text descriptions
    df_global["clindiag__text"] = df_global["clindiag__decod"].map(clindiag_mapping)

    # Create a histogram using the text values
    plt.figure(figsize=(20, 17))
    df_global["clindiag__text"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Histogram of clindiag__decod")
    plt.xlabel("Clinical Diagnosis")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot if saved_result_path is provided
    if saved_result_path:
        my_file = "Histogram_clindiag_decod.png"
        plt.savefig(os.path.join(saved_result_path, my_file), bbox_inches="tight")

    # Show the plot
    plt.show()

    """
    CLASSIFICATION PLOTS
    """


def plot_confusion_matrix(y_true, y_pred, file_name):
    conf_mat = confusion_matrix(y_true, y_pred)
    class_labels = ["mtDNA", "nDNA"]
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = [f"{value:0.0f}" for value in conf_mat.flatten()]
    labels = np.asarray(
        [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    ).reshape(2, 2)

    sns_plot = sns.heatmap(
        conf_mat, annot=labels, fmt="", cmap="Blues", annot_kws={"size": 18}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=[0.5, 1.5], labels=class_labels)
    plt.yticks(ticks=[0.5, 1.5], labels=class_labels)
    figure = plt.gcf()
    figure.set_size_inches(7, 6)

    plt.savefig(
        os.path.join(saved_result_path_classification, f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
    )
    plt.close()


def plot_top_feature_importance(importances, names, model_type, save_path, top_n=10):
    feature_importance = np.array(importances)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = (
        pd.DataFrame(data)
        .sort_values(by="feature_importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="feature_importance", y="feature_names", data=fi_df)
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.title(f"Top {top_n} Features - {model_type}")

    my_file = f"Top_{top_n}_Feature_importance_{model_type}.png"
    plt.savefig(os.path.join(save_path, my_file), bbox_inches="tight")
    plt.close()


def plot_shap_values(
    estimator,
    X,
    results_path,
    model_name,
    feature_selection_option,
    balancing_technique,
):
    explainer = shap.Explainer(estimator, X)
    shap_values = explainer(X)
    shap.plots.bar(shap_values)
    shap_bar_plot_file = f"shap_bar_plot_{model_name}_{feature_selection_option}_{balancing_technique}.png"
    plt.savefig(
        os.path.join(results_path, shap_bar_plot_file),
        format="png",
        bbox_inches="tight",
    )
    plt.close()
