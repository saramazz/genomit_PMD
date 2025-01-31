"""
Script to perform classification of nDNA vs mtDNA.
Evaluates the BEST classifier
"""

# Standard Library Imports
import os
import sys
import time
import json
from collections import Counter
from itertools import combinations
from datetime import datetime
import joblib
import os

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

# Constants and Paths
GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
BEST_PATH = os.path.join(saved_result_path_classification, "best_model")
EXPERIMENT_PATH = os.path.join(
    saved_result_path_classification, "experiments_all_models"
)
VERSION = "20250128_165035"  # best model version

# Ensure necessary directories exist
os.makedirs(BEST_PATH, exist_ok=True)


def load_and_prepare_data():
    """Load and prepare the DataFrame for classification."""
    if not os.path.exists(GLOBAL_DF_PATH):
        raise FileNotFoundError(f"File not found: {GLOBAL_DF_PATH}")

    df = pd.read_csv(GLOBAL_DF_PATH)

    # Display initial DataFrame information
    nRow, nCol = df.shape
    print(f'The DataFrame "df_preprocessed" contains {nRow} rows and {nCol} columns.')

    # print("Columns:", df.columns)

    # Preprocess target column, handle NaNs, and print DataFrame
    df, df_not_numerical = process_gendna_column(df)
    df = fill_missing_values(df)

    # Fill the remaining missing values with 998
    df = df.fillna(998)

    # Check if there are any remaining missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(
            f"Warning: There are still {missing_count} missing values in the DataFrame."
        )
    else:
        print("All missing values have been successfully filled.")

    return df


def feature_selection(df):
    """Select and drop unnecessary features from the DataFrame."""
    df_vars = pd.read_excel(important_vars_path)
    column_name = "consider for mtDNA vs nDNA classification?"
    columns_to_drop = list(df_vars.loc[df_vars[column_name] == "N", "variable"])

    # Additional columns to drop
    additional_columns = [
        "Hospital",
        "nDNA",
        "mtDNA",
        "gendna_type",
        "epiphen",
        "sll",
        "clindiag__decod",
        "encephalopathy",
    ]
    additional_columns += [
        col for col in df.columns if "pimgtype" in col or "psterm" in col
    ]

    columns_to_drop += additional_columns
    # print("Columns to drop:", columns_to_drop)

    X, y = define_X_y(df, columns_to_drop)
    return X, y


def print_data_info(X_train, X_test, y_train, y_test, features, df):
    """Prints information about the datasets and features."""
    # print the experiment path
    print("Saving path:", BEST_PATH)

    print("Dataset for classification shape:", df.shape)

    print("Dimension of X_train:", X_train.shape)
    print("Dimension of X_test:", X_test.shape)
    print("Dimension of y_train:", y_train.shape)
    print("Dimension of y_test:", y_test.shape)
    print("Features names:", features)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""
    # file_name = f"classification_reports_{current_datetime}_no_mrmr.txt"
    file_name = f"classification_reports_best.txt"
    sys.stdout = open(os.path.join(BEST_PATH, file_name), "w")


def perform_classification_best(X_test, y_test, clf_model, results_path, features):
    '''
    Function to predict on the test set and print the classification results and plots
    '''

    # Evaluate on the test set
    y_pred = clf_model.predict(X_test)

    # Calculate performance metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Check if the model has feature importances
    if hasattr(clf_model, "feature_importances_"):
        importances = clf_model.feature_importances_
        feature_importances = {features[i]: importances[i] for i in range(len(importances))}
        indices_all = np.argsort(importances)  # Sort indices by importance
        feature_importance_data = {
            "feature_importances": feature_importances,
            "top_10_features": {features[i]: importances[i] for i in indices_all[-10:]},
        }
    else:
        importances = np.array([])  # handle cases where feature importances are not available
        feature_importance_data = {}

    # Save all relevant results to a file
    results_to_save = {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_pred": y_pred,
        "y_test": y_test,
        "importances": importances,
        "feature_importance_data": feature_importance_data,
    }

    # Create results directory if needed and save pickled results
    os.makedirs(results_path, exist_ok=True)
    results_file_path = os.path.join(results_path, "best_model_results.pkl")
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)

    print(f"Results saved to {results_file_path}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")

    # Plot Confusion Matrix
    print("Plotting Confusion Matrix...")
    confusion_matrix_file = "cm_best_model.png"
    plot_confusion_matrix(y_test, y_pred, os.path.join(results_path, confusion_matrix_file))
    plt.close()

    # Plot Importances if available
    if importances.size > 0:
        print("Calculating and Plotting Importances...")
        plt.figure(figsize=(10, 8))
        plt.title("All feature Importances", fontsize=15)
        plt.barh(range(len(indices_all)), importances[indices_all], color="lightblue", align="center")
        plt.yticks(range(len(indices_all)), [features[i] for i in indices_all], ha="right", fontsize=10)
        plt.xlabel("Relative Importance", fontsize=15)
        feature_importance_file = "feature_imp_best_ALL.png"
        plt.savefig(os.path.join(results_path, feature_importance_file), format="png", bbox_inches="tight")
        plt.close()

        # Plot ONLY top 10 feature importances
        plt.figure(figsize=(10, 8))
        plt.title("Top 10 Feature Importances", fontsize=15)
        plt.barh(range(len(indices_all[-10:])), importances[indices_all[-10:]], color="lightblue", align="center")
        plt.yticks(range(len(indices_all[-10:])), [features[i] for i in indices_all[-10:]], ha="right", fontsize=10)
        plt.xlabel("Relative Importance", fontsize=15)
        feature_importance_file = "feature_imp_best_10.png"
        plt.savefig(os.path.join(results_path, feature_importance_file), format="png", bbox_inches="tight")
        plt.close()

    # Plot SHAP Bar plot if possible
    if hasattr(clf_model, "predict_proba"):
        try:
            explainer = shap.Explainer(clf_model, X_test)
            shap_values = explainer(X_test)
            shap.plots.bar(shap_values)
            shap_bar_plot_file = "shap_bar_plot_best.png"
            plt.savefig(os.path.join(results_path, shap_bar_plot_file), format="png", bbox_inches="tight")
            plt.close()

        except Exception as e:
            print("Error plotting SHAP bar plot:", str(e))

    return results_to_save


def main():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_output(current_datetime)
    # Load and preprocess data
    df = load_and_prepare_data()
    X, y = feature_selection(df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        test_subjects,
        features,
        kf,
        scorer,
        thr,
        nFeatures,
        num_folds,
    ) = experiment_definition(X, y, df, EXPERIMENT_PATH)

    #Save the df to a pickle file and to a csv file
    df.to_pickle(os.path.join(BEST_PATH, "df_classification.pkl"))
    df.to_csv(os.path.join(BEST_PATH, "df_classification.csv"))



    #save the features to a text list
    with open(os.path.join(BEST_PATH, "features.txt"), "w") as f:
        for item in features:
            f.write("%s\n" % item)
            

    print_data_info(
        X_train, X_test, y_train, y_test, features, df.drop(columns=["subjid"])
    )

    print("Starting the classification...")

    best_model_path = os.path.join(EXPERIMENT_PATH, f"best_classifier_{VERSION}.pkl")
    with open(best_model_path, "rb") as f:
        best_classifiers = pickle.load(f)

    # If the tuple has a consistent format as expected
    if isinstance(best_classifiers, tuple):
        classifier_name, best_estimator, best_score, results = best_classifiers[:4]
        best_params = results["best_params"]
        print(f"Classifier Name: {classifier_name}")
        print(f"Best Estimator: {best_estimator}")
        print(f"Best Score: {best_score}")
        print("Best Parameters:", best_params)

    # Create an imputer object with 'constant' strategy and your specified fill value
    imputer = SimpleImputer(strategy="constant", fill_value=998)

    # Fit the imputer on the training data and transform it
    X_train = imputer.fit_transform(X_train)
    # Transform the test data with the same imputer
    X_test = imputer.transform(X_test)

    perform_classification_best(X_test, y_test, best_estimator, BEST_PATH, features)

    print("Classification with the best classifier completed and results saved.")

if __name__ == "__main__":
    main()

"""
# Define the classifiers and their parameter grids

classifiers = {
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": range(2),
            "n_estimators": range(5),
            "learning_rate": [0.2],
        },
    ),
}

# define the settings of the experiment
balancing_techniques = [
    "under",
]
feature_selection_options = ["no"]

'''

classifiers = {
    "XGBClassifier": (
        XGBClassifier(),
        {
            "max_depth": range(2, 22, 4),
            "n_estimators": range(50, 500, 50),
            "learning_rate": [0.01, 0.1, 0.2],
        },
    ),
    "DecisionTreeClassifier": (
        DecisionTreeClassifier(),
        {
            "max_depth": range(2, 30, 2),
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2", None],
        },
    ),
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
    ),
    "SVM": (
        SVC(),
        {
            "C": [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
            "kernel": ["rbf"],
        },
    ),
}

# define the settings of the experiment
balancing_techniques = [
    "no",
    "over",
    "under",
]
feature_selection_options = ["no", "pca", "mrmr", "select_from_model", "rfe"]

"""


"""
for classifier, (clf_model, param_grid) in classifiers.items():
    print("********************************************************************\n")
    pipeline = Pipeline([("clf", clf_model)])  # create pipeline to clf

    for feature_selection_option in feature_selection_options:
        # Apply feature selection option to the data if necessary
        # Process feature selection
        X_train_selected, X_test_selected, param_grid_selected, pipeline_selected = (
            process_feature_selection(
                clf_model,
                df#X_df,
                X_train,
                X_test,
                y_train,
                param_grid,
                pipeline,
                scorer,
                kf,
                feature_selection_option,
                features,
                num_folds,
                nFeatures,
                thr,
            )
        )

        for balancing_technique in balancing_techniques:
            # print the current classification settings in one line only
            print("------------------------------------------------------------\n")
            print(
                f"Classifier: {classifier}, Balancing Technique: {balancing_technique}, Feature Selection Option: {feature_selection_option}"
            )

            # Apply balancing technique to the data if necessary
            X_train_bal, y_train_bal, X_test_bal, y_test_bal = balance_data(
                X_train_selected, y_train, X_test_selected, y_test, balancing_technique
            )

            # finally, perform the classification
            perform_classification(
                clf_model,
                param_grid,
                pipeline_selected,
                df, #X_df
                X_train_bal,
                X_test_bal,
                y_train_bal,
                y_test_bal,
                kf,
                scorer,
                features,
                balancing_technique,
                feature_selection_option,
                saved_result_path_classification_exp,
            )


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__



# Read the text file
with open(os.path.join(saved_result_path_classification, file_name), 'r') as file:
    lines = file.readlines()

# Initialize lists to store data
classifiers = []
balancing_techniques = []
feature_selection_options = []
weighted_avg_f1_scores = []
accuracies = []
f1_score_mt = []
f1_score_wt = []

# Parse the text file and extract data
for line in lines:
    if line.startswith("Classifier"):
        classifiers.append(line.split(":")[1].strip().split(',')[0].strip())
        balancing_techniques.append(line.split(":")[2].strip().split(',')[0].strip())
        feature_selection_options.append(line.split(":")[3].strip().split(',')[0].strip())
    elif line.startswith("weighted avg"):
        weighted_avg_f1_scores.append(float(line.split()[3]))
        accuracies.append(float(line.split()[2]))
    elif line.startswith("f1-score_mt"):
        f1_score_mt.append(float(line.split()[1]))
        f1_score_wt.append(float(line.split()[3]))

# Create a DataFrame
df = pd.DataFrame({
    'Classifier': classifiers,
    'Balancing Technique': balancing_techniques,
    'Feature Selection Option': feature_selection_options,
    'Weighted Avg F1-Score': weighted_avg_f1_scores,
    'Accuracy': accuracies,
    'F1-Score MT': f1_score_mt,
    'F1-Score WT': f1_score_wt
})

print("Classification results:")
print(df)

# Save the DataFrame to an Excel file
df.to_excel('classification_output.xlsx', index=False)
"""
