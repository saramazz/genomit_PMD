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
# VERSION = "20250210_164401"  # best model version
VERSION = "20250212_144836"

#ask if consider patients with no sympthoms
Input=input("Do you want to consider patients with no symptoms? (y/n)")
if Input=="y":
    GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed_all.csv")
    EXPERIMENT_PATH = os.path.join(
        saved_result_path_classification, "experiments_all_models_all"
    )
    BEST_PATH = os.path.join(saved_result_path_classification, "best_model_all")
    VERSION = "20250217_122843"

#Print best path and experiment path
print("Best path: ", BEST_PATH)
print("Experiment path: ", EXPERIMENT_PATH)

    

# Ensure necessary directories exist
os.makedirs(EXPERIMENT_PATH, exist_ok=True)





# Ensure necessary directories exist
os.makedirs(BEST_PATH, exist_ok=True)


def setup_output(current_datetime):
    """Set up output redirection to a log file."""
    # file_name = f"classification_reports_{current_datetime}_no_mrmr.txt"
    # ask if clf or res
    choice = input("Press clf to start the classification or res to display the res...")
    if choice == "clf":
        file_name = f"classification_reports_best.txt"
    else:
        file_name = f"classification_analysis_results.txt"
    sys.stdout = open(os.path.join(BEST_PATH, file_name), "w")


def perform_classification_best(X_test, y_test, clf_model, results_path, features):
    """
    Function to predict on the test set and print the classification results and plots
    """

    # Evaluate on the test set
    y_pred = clf_model.predict(X_test)

    # Calculate performance metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Check if the model has feature importances
    if hasattr(clf_model, "feature_importances_"):
        importances = clf_model.feature_importances_
        feature_importances = {
            features[i]: importances[i] for i in range(len(importances))
        }
        indices_all = np.argsort(importances)  # Sort indices by importance
        feature_importance_data = {
            "feature_importances": feature_importances,
            "top_10_features": {features[i]: importances[i] for i in indices_all[-10:]},
        }
    else:
        importances = np.array(
            []
        )  # handle cases where feature importances are not available
        feature_importance_data = {}

    # Save all relevant results to a file
    results_to_save = {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
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
    plot_confusion_matrix(
        y_test, y_pred, os.path.join(results_path, confusion_matrix_file)
    )
    plt.close()

    # Plot Importances if available
    if importances.size > 0:
        print("Calculating and Plotting Importances...")
        plt.figure(figsize=(10, 8))
        plt.title("All feature Importances", fontsize=15)
        plt.barh(
            range(len(indices_all)),
            importances[indices_all],
            color="lightblue",
            align="center",
        )
        plt.yticks(
            range(len(indices_all)),
            [features[i] for i in indices_all],
            ha="right",
            fontsize=10,
        )
        plt.xlabel("Relative Importance", fontsize=15)
        feature_importance_file = "feature_imp_best_ALL.png"
        plt.savefig(
            os.path.join(results_path, feature_importance_file),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

        # Plot ONLY top 10 feature importances
        plt.figure(figsize=(10, 8))
        plt.title("Top 10 Feature Importances", fontsize=15)
        plt.barh(
            range(len(indices_all[-10:])),
            importances[indices_all[-10:]],
            color="lightblue",
            align="center",
        )
        plt.yticks(
            range(len(indices_all[-10:])),
            [features[i] for i in indices_all[-10:]],
            ha="right",
            fontsize=10,
        )
        plt.xlabel("Relative Importance", fontsize=15)
        feature_importance_file = "feature_imp_best_10.png"
        plt.savefig(
            os.path.join(results_path, feature_importance_file),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

    # Plot SHAP Bar plot if possible
    if hasattr(clf_model, "predict_proba"):
        try:

            # Setting matplotlib to use the 'Agg' backend
            plt.switch_backend("Agg")

            # Create SHAP explainer and compute SHAP values
            explainer = shap.Explainer(clf_model, X_test)
            shap_values = explainer(X_test)

            # Generate the SHAP bar plot
            shap.plots.bar(shap_values)

            # Define file path and save the plot
            shap_bar_plot_file = "shap_bar_plot_best.png"
            plt.savefig(
                os.path.join(results_path, shap_bar_plot_file),
                format="png",
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"SHAP bar plot saved at {os.path.join(results_path, shap_bar_plot_file)}"
            )

        except Exception as e:
            print("Error plotting SHAP bar plot:", str(e))

    return results_to_save


def main():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_output(current_datetime)
    # Load and preprocess data
    df, mt_DNA_patients = load_and_prepare_data(GLOBAL_DF_PATH)

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
    ) = experiment_definition(X, y, df, EXPERIMENT_PATH, mt_DNA_patients)

    # Save the df to a pickle file and to a csv file
    df.to_pickle(os.path.join(BEST_PATH, "df_classification.pkl"))
    df.to_csv(os.path.join(BEST_PATH, "df_classification.csv"))

    # save the features to a text list
    with open(os.path.join(BEST_PATH, "features.txt"), "w") as f:
        for item in features:
            f.write("%s\n" % item)

    print_data_info(
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        df.drop(columns=["subjid"]),
        EXPERIMENT_PATH,
    )

    print("Starting the classification...")
    '''


    #if using the best model file
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
    '''


    

    classifiers={
                "XGBClassifier":(XGBClassifier(),
            {
                "max_depth": [6],  # Most impactful depths
                "n_estimators": [50],  # Reduced number of trees
                "learning_rate": [0.1],  # Wide range learning rates
                "subsample": [1.0],  # Essential variations
                "colsample_bytree": [0.8],  # Stable choice
                "reg_alpha": [0],  # Skip regularization for complexity
                "reg_lambda": [1],  # Default value
            })}
    


  # Initialize GridSearchCV with the pipeline or model
    for classifier, (clf_model, param_grid) in classifiers.items():
        """adding the selector"""

        
        selector = RFE(estimator=clf_model)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        # print the dimension of the X_train and X_test
        print("Dimension of X_train and X_test")
        print(X_train.shape)
        print(X_test.shape)


        # Rimuove la colonna 'subjid' se esiste in X_df
        if "subjid" in df.columns:
            df = df.drop(columns=["subjid"])

        selected_features = df.columns[selector.get_support()]
        print("Selected Features:")
        print(selected_features)
        print("Number of Selected Features:", len(selected_features))

        '''
        # features selected
        selected_features = ['aao', 'alcohol', 'bgc', 'bmi', 'card_abn', 'ear_voice_abn', 'echo',
        'eeg', 'eye_abn', 'fhx', 'gait_abn', 'hba1c', 'hisc', 'internal_abn',
        'lac', 'le', 'mhterm', 'mrisll', 'musc_sk_abn', 'pcgntn', 'pssev',
        'resp', 'smoking', 'symp_on_1', 'symp_on_3', 'symp_on_4', 'va', 'wml']
        print("Features used in the classification:")
        print(selected_features)
        '''
        # save the selected features to a text file
        with open(os.path.join(BEST_PATH, "selected_features.txt"), "w") as f:
            for item in selected_features:
                f.write("%s\n" % item)


        '''
        # Assuming column names available, construct DataFrame
        X_test = pd.DataFrame(
            X_test, columns=features
        )  # 'all_column_names' should be defined

        # Select features
        X_test = X_test[selected_features]

        X_train = pd.DataFrame(X_train, columns=features)
        X_train = X_train[selected_features]
        '''


        oversampler = SMOTE(random_state=42)
        X_train, y_train = oversampler.fit_resample(
            X_train, y_train
        )
        X_test, y_test = (
            X_test,
            y_test,
        )  # No resampling for test data

        


        #print dimension of X_train and X_test
        print("Dimension of X_train and X_test")
        print(X_train.shape)
        print(X_test.shape)

  
        grid_search = GridSearchCV(
            estimator=clf_model,
            param_grid=param_grid,
            cv=kf,  # kf, #TODO PUT IT on kf
            scoring=scorer,
            verbose=1,  # TODO put on 1
            n_jobs=-1,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)

        # Extract results
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        best_score_ = (
            grid_search.best_score_
        )  # performance metric of the best_estimator on the train set
        cv_results = grid_search.cv_results_

        print("Training performances:")
        print("Best score:", best_score_)
        print("Best params:", best_params)      


        perform_classification_best(
            X_test, y_test, best_estimator, BEST_PATH, selected_features
        )


    print("Classification with the best classifier completed and results saved.")

    # Load the results and print them
    results_file_path = os.path.join(BEST_PATH, "best_model_results.pkl")
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)

    print("Results:")
    print(results)

    # read the feature importance data in the results
    feature_importance_data = results["feature_importance_data"]
    print("Feature Importance Data:")
    print(feature_importance_data)

    # print the distribution of the best 10 features important
    top_10_features = feature_importance_data["top_10_features"]
    print("Top 10 Features:")
    print(top_10_features)
    # print the distribution of the best 10 features important in a bar plot from the test set
    # identify the test subjects in the df raw
    df_raw = pd.read_csv(GLOBAL_DF_PATH)
    df_raw_test = df_raw[df_raw["subjid"].isin(test_subjects)]

    # create folder for the plots
    os.makedirs(os.path.join(BEST_PATH, "TOP_feature_distributions"), exist_ok=True)
    for col in top_10_features.keys():
        sns.histplot(df_raw_test[col], kde=True)
        plt.title(f"Distribution of {col}")
        # save it to the folder
        plt.savefig(
            os.path.join(
                BEST_PATH, "TOP_feature_distributions", f"{col}_distribution.png"
            )
        )
        plt.close()
        # print the distribution of the feature
        print(f"Feature: {col}")
        print(df_raw_test[col].describe())
        print("Distribution:")
        print(df_raw_test[col].value_counts())
        print("-----------------------------------")
        print("\n")
    # print that the plots are saved
    print("Top 10 Feature Distributions saved in the folder TOP_feature_distributions")


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
        X_train, X_test, param_grid_selected, pipeline_selected = (
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
                X_train, y_train, X_test, y_test, balancing_technique
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
