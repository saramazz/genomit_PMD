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


"""
Functions to plot for preprocessing and classification
"""

"""
PLOT PREPROCESSING
"""


def plot_missing_values(df, saved_result_path):
    missing_values = df.isnull().sum()
    total_values = len(df)
    missing_percentages = (missing_values / total_values) * 100
    missing_data = pd.DataFrame(
        {"Missing Values": missing_values, "Missing Percentage": missing_percentages}
    )
    missing_data = missing_data.sort_values(by="Missing Values", ascending=False)
    plt.figure(figsize=(12, 18))
    plt.barh(missing_data.index, missing_data["Missing Values"], color="lightcoral")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Variable Name")
    plt.title("Missing Values for Each Variable")
    plt.gca().invert_yaxis()
    # Decrease the font size on the y-axis
    plt.yticks(fontsize=10)
    plt.grid()
    my_file = "Histogram_MissingValues_df_Global.png"
    plt.savefig(os.path.join(saved_result_path, my_file), bbox_inches="tight")
    # print that has been saved in the path
    print(f"Missing values plot saved in {saved_result_path}")
    # plt.show()
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
    Plot the distribution of 'nDNA' and 'mtDNA' classes in a pie chart.

    Parameters:
    - df: DataFrame containing 'nDNA' and 'mtDNA' columns.
    - save_path: Path to save the generated plot.
    """
    # Set font size for all components
    plt.rcParams["font.size"] = 20

    # Use seaborn color palette for shades of blue
    colors = sns.color_palette("Blues", len(df[["nDNA", "mtDNA"]].columns))

    # Calculate class counts and total samples
    class_counts = df[["nDNA", "mtDNA"]].apply(pd.Series.value_counts).sum()
    total_samples = len(df[["nDNA", "mtDNA"]])

    # Plot the pie chart
    plt.figure(figsize=(15, 8))
    class_counts.plot.pie(
        autopct=lambda p: f"{p:.1f}%",
        figsize=(15, 8),
        labels=class_counts.index,
        legend=True,
        colors=colors,
    )

    # Set title and legend
    plt.title("Distribution of nDNA vs mtDNA Classes", fontsize=20)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=20)

    # Save the plot
    my_file = "gendna_distribution_pie_n_mt.png"
    plt.savefig(
        os.path.join(saved_result_path_classification, my_file), bbox_inches="tight"
    )
    print(f"gendna distribution plot saved in {os.path.join(saved_result_path_classification, my_file)}")
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


# Function to plot the feature importance
def plot_top_feature_importance(importances, names, model_type, save_path, top_n=10):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importances)
    feature_names = np.array(names)

    print(len(feature_importance))
    print(len(feature_names))

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Select the top N features
    fi_df = fi_df.head(top_n)

    # Define the size of the bar plot
    plt.figure(figsize=(10, 8))

    # Plot Seaborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    # sns.set(font_scale=2)

    # Add chart labels
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.title(f"Top {top_n} Features - {model_type}")

    # Save the plot
    my_file = f"Top_{top_n}_Feature_importance_{model_type}.png"
    plt.savefig(
        os.path.join(saved_result_path_classification, my_file), bbox_inches="tight"
    )


def precision_recall_plot(gt, pp, title, file_name, global_path):
    # Set the font sizes for various parts of the plot
    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (15, 5),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)

    # Calculate precision and recall
    probs_rf1 = pp[:, 1]
    probs_rf0 = pp[:, 0]
    precision_rf1, recall_rf1, _ = precision_recall_curve(gt, probs_rf1)
    ap_rf1 = average_precision_score(gt, probs_rf1)
    precision_rf0, recall_rf0, _ = precision_recall_curve(1 - gt, probs_rf0)
    ap_rf0 = average_precision_score(1 - gt, probs_rf0)

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(recall_rf1, precision_rf1, label=f"AP (0) = {ap_rf1:.2f}")
    plt.plot(recall_rf0, precision_rf0, label=f"AP (1) = {ap_rf0:.2f}")
    # plt.title(title, size=27)
    plt.xlabel("Recall", size=30)
    plt.ylabel("Precision", size=30)
    plt.legend(loc="best")

    # Save the plot
    my_file = file_name + ".png"
    saved_result_path_classification = os.path.join(global_path, "saved_results")
    plt.savefig(
        os.path.join(saved_result_path_classification, my_file),
        format="png",
        bbox_inches="tight",
    )


def plot_confusion_matrix(y_true, y_pred, file_name):
    # Confusion matrix
    # conf_mat = metrics.confusion_matrix(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Define class labels
    class_labels = ["mtDNA", "nDNA"]
    # df_non_nan['gendna_type'] = df_non_nan['gendna_type'].replace({'mtDNA': 0, 'nDNA': 1})

    # Plot confusion matrix
    confusion_matrix_plot(conf_mat, class_labels, file_name)


def confusion_matrix_plot(conf_mat, class_labels, file_name):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    sns_plot = sns.heatmap(
        conf_mat, annot=labels, fmt="", cmap="Blues", annot_kws={"size": 18}
    )

    # Set x and y axis labels
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Set x and y axis ticks
    plt.xticks(ticks=[0.5, 1.5], labels=class_labels)
    plt.yticks(ticks=[0.5, 1.5], labels=class_labels)

    # Set title
    # plt.title(title)

    figure = plt.gcf()
    figure.set_size_inches(7, 6)

    my_file = file_name + ".png"
    plt.savefig(
        os.path.join(saved_result_path_classification, my_file),
        format="png",
        bbox_inches="tight",
    )
    plt.close()
