"""
Functions to do classification
"""

# Standard library imports
import os
from collections import Counter
from itertools import combinations

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap  # for SHAP values

# Imbalanced-learn library imports
from imblearn.over_sampling import SMOTE

# import ADASYN
from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import RandomForestClassifier  # For Random Forest
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree
from sklearn.svm import SVC  # For Support Vector Machines (SVM)

# Scikit-learn library imports
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    precision_recall_curve,
    mean_squared_error,
)
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    mutual_info_classif,
    SequentialFeatureSelector,
    SelectFromModel,
    RFE,
)
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

# Gradient boosting and boosting classifier imports
from xgboost import XGBClassifier

# MRMR for feature selection
import mrmr


# Local imports
from config import (
    global_path,
    saved_result_path,
    saved_result_path_classification,
    important_vars_path,
)
from utilities import *
from plotting import *
from preprocessing import *

# import PCA
from sklearn.decomposition import PCA


# %% Feature ranking on the training set using cross-validation
def rankfeatures(X_train, Y_train, frmethod, nFolds, nFeatures, thr, kf):
    Y_train = np.asarray(Y_train)
    top_features_all = []

    # Generate shuffled patient indices
    patient_indices = np.arange(X_train.shape[0])
    np.random.seed(42)
    np.random.shuffle(patient_indices)

    # Process K-fold splitting
    for fold, (train_indices, val_indices) in enumerate(kf.split(patient_indices)):
        print(f"Processing Fold {fold + 1}")
        X_train_cv = X_train[train_indices]
        y_train_cv = Y_train[train_indices]

        if frmethod == "mrmr":
            print(f"MRMR feature ranking - Fold {fold + 1} of {nFolds}")
            top_feature_indices = mrmr.mrmr_regression(
                X=pd.DataFrame(X_train),
                y=pd.DataFrame(Y_train),
                K=int(thr * X_train.shape[1]),
            )
            top_features_all.append(top_feature_indices)

    # Process feature aggregation
    top_features_all_ = np.asarray(top_features_all, dtype=np.float32)
    print("Found top_features_all_", top_features_all_)

    all_top_features = np.unique(top_features_all_)
    feature_scores = np.empty([len(all_top_features), 2], dtype=int)

    for ii, ff in enumerate(all_top_features):
        idx_ff = np.asarray(np.where(np.isin(top_features_all_, ff)))
        score = np.sum(nFeatures - idx_ff[1])
        feature_scores[ii, :] = [ff, score]

    sorted_scores = np.argsort(feature_scores[:, 1])[::-1]
    top_features = feature_scores[sorted_scores[: int(thr * X_train.shape[1])], 0]

    print("Top 25% Features:")
    print(top_features)
    print("Number of Top Features:", len(top_features))

    return top_features


# Define a function to calculate the distribution of a categorical column
def calculate_distribution(df, column_name, mapping):
    # Map values to labels
    df[f"{column_name}_label"] = df[column_name].map(mapping)

    # Calculate distribution
    distribution = (
        df[f"{column_name}_label"].value_counts(dropna=True) / len(df)
    ).to_dict()

    # Round and update the distribution based on the mapping
    distribution = {
        label: round(distribution.get(label, 0), 4) for label in mapping.values()
    }
    return distribution


# Define a mapping function to assign 'gendna_type'
def assign_gendna_type(value):
    if value in [4, 6, 8]:
        return "nDNA"
    elif value in [5, 7]:
        return "mtDNA"
    else:
        # print the value
        # print("Value of gendna: ", value)
        return "Unknown"  # Use an explicit category or designate a special case


def process_gendna_column(df):
    """
    Process the 'gendna' column in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the 'gendna' column.
        saved_result_path_classification (str): The path to save the result files.

    Returns:
        DataFrame: The DataFrame with processed 'gendna' column.
    """

    # Check if there are any remaining missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(
            f"Warning: There are still {missing_count} missing values in the DataFrame."
        )
    else:
        print("All missing values have been successfully filled.")

    # check if there are patients with nan values in the gendna column
    if df["gendna"].isnull().sum() > 0:
        print("There are patients with NaN values in the 'gendna' column.")
        # Column to be processed
        column = "gendna"

        # Display distribution of values in the 'gendna' column
        # print(f"Distribution for column: {column}")
        # print(df[column].value_counts())

        # Count the NaN values in the 'gendna' column
        nan_count = df["gendna"].isna().sum()
        # print(f"Number of NaN values in '{column}': {nan_count}")

        # Save patients with NaN 'gendna' information to a file
        file_path = os.path.join(
            saved_result_path_classification, "patients_with_nan_or_1_gendna.csv"
        )
        patients_with_nan_gendna = df[df["gendna"].isnull()][["subjid", "Hospital"]]
        # add to the patients_with_nan_gendna the patients where gendna is equal to 1
        patients_with_1_gendna = df[df["gendna"] == 1][["subjid", "Hospital"]]
        patients_with_nan_gendna = pd.concat(
            [patients_with_nan_gendna, patients_with_1_gendna]
        )
        # create a df with the all the columns of these patients
        patients_with_nan_gendna = df[
            df["subjid"].isin(patients_with_nan_gendna["subjid"])
        ]
        patients_with_nan_gendna.to_csv(file_path, index=False)
        # convert to numerical and save it patients_with_nan_gendna
        patients_with_nan_gendna = convert_to_numerical(patients_with_nan_gendna)
        patients_with_nan_gendna.to_csv(
            os.path.join(
                saved_result_path_classification,
                "patients_with_nan_or_1_gendna_num.csv",
            ),
            index=False,
        )
        # print the dimension of the file
        # print("Patients with NaN or 1 'gendna' information saved to:", file_path)
        # print("Dimension of the file:", patients_with_nan_gendna.shape)

        # Drop NaN values from 'gendna' column
        df_non_nan = df.dropna(subset=["gendna"])

    df_processed = df.copy()

    # Remove rows where 'gendna' is equal to 1
    df_processed = df_processed[df_processed["gendna"] != 1]

    # Assign 'gendna_type' based on existing 'gendna' values
    df_processed["gendna_type"] = df_processed["gendna"].apply(assign_gendna_type)

    # Print distribution of 'gendna_type'
    print("\nDistribution of 'gendna_type':")
    print(df_processed["gendna_type"].value_counts())

    # Convert 'gendna_type' into numerical values
    # Here we assume "Unknown" is not required further; adjust as needed
    df_processed["gendna_type_num"] = df_processed["gendna_type"].replace(
        {
            "mtDNA": 0,
            "nDNA": 1,
            # Optionally, you might also handle "Unknown" here if it remains
        }
    )
    # remove Unknown patienta
    df_processed = df_processed[df_processed["gendna_type"] != "Unknown"]
    # print how many unknown patients were removed
    print("Number of Unknown patients removed: ", len(df) - len(df_processed))

    # Plot distribution of 'gendna_type'
    plot_gendna_distribution(df_processed)

    # Convert DataFrame to numerical format
    df_processed = convert_to_numerical(df_processed)

    print("\nDistribution of 'gendna_type' in the numerical DataFrame:")
    print(df_processed["gendna_type"].value_counts())
    # print as percentage
    print("\nPercentage distribution of 'gendna_type' in the numerical DataFrame:")
    print(df_processed["gendna_type"].value_counts(normalize=True) * 100)

    # Check if there are any remaining missing values
    missing_count = df_processed.isnull().sum().sum()
    if missing_count > 0:
        print(
            f"Warning: There are still {missing_count} missing values in the DataFrame."
        )
    else:
        print("All missing values have been successfully filled.")

    return df_processed


def add_patients_to_reach_179(test_subjects_ids, df, mt_DNA_patients):
    total_required = 179
    mtDNA_required = 110
    nDNA_required = 69

    # print df columns
    # add gendna_type column using the mt_DNA_patients
    # extract values of mtDNA

    df["gendna_type"] = df["subjid"].apply(
        lambda x: 0 if x in mt_DNA_patients["subjid"].values else 1
    )

    # Current counts
    current_mtDNA_count = len(
        df[(df["subjid"].isin(test_subjects_ids)) & (df["gendna_type"] == 0)]
    )
    current_nDNA_count = len(
        df[(df["subjid"].isin(test_subjects_ids)) & (df["gendna_type"] == 1)]
    )

    # Calculate the missing numbers to reach the required totals
    missing_mtDNA = mtDNA_required - current_mtDNA_count
    missing_nDNA = nDNA_required - current_nDNA_count

    if missing_mtDNA <= 0 and missing_nDNA <= 0:
        print("No additional patients are needed.")
        return test_subjects_ids

    # Filter available patients who are not already in test_subjects_ids
    available_mtDNA_patients = df[
        (df["gendna_type"] == 0) & (~df["subjid"].isin(test_subjects_ids))
    ]
    available_nDNA_patients = df[
        (df["gendna_type"] == 1) & (~df["subjid"].isin(test_subjects_ids))
    ]

    # Check for availability of required patients
    if missing_mtDNA > len(available_mtDNA_patients):
        raise ValueError(
            "Not enough new mtDNA patients to reach the required count of 110."
        )
    if missing_nDNA > len(available_nDNA_patients):
        raise ValueError(
            "Not enough new nDNA patients to reach the required count of 69."
        )

    new_mtDNA_patients = available_mtDNA_patients["subjid"].sample(
        n=missing_mtDNA, random_state=42
    )
    new_nDNA_patients = available_nDNA_patients["subjid"].sample(
        n=missing_nDNA, random_state=42
    )

    # Add the new patients to test_subjects_ids
    updated_test_subjects_ids = (
        test_subjects_ids + new_mtDNA_patients.tolist() + new_nDNA_patients.tolist()
    )

    print(
        f"Added {len(new_mtDNA_patients)} mtDNA patients and {len(new_nDNA_patients)} nDNA patients. "
        f"Total mtDNA: {current_mtDNA_count + len(new_mtDNA_patients)}, Total nDNA: {current_nDNA_count + len(new_nDNA_patients)}."
    )

    # remove df["gendna_type"]
    df = df.drop(columns=["gendna_type"])

    return updated_test_subjects_ids


def experiment_definition(X, y, X_df, saving_path, mt_DNA_patients, num_folds=5):

    classifier_config_path = os.path.join(saving_path, "classifier_configuration.pkl")

    if not os.path.exists(classifier_config_path):
        print("Classifier configuration does not exist. Creating the configuration...")

        #consider the column test of the df and filter the test subjects
        test_subjects_ids = X_df[X_df["test"] == 1]["subjid"].values

        #remove the test column
        X_df = X_df.drop(columns=["test"])
        
        # print the length of test_subjects_ids 
        print("Length of test_subjects_ids: ", len(test_subjects_ids))
        
        input("Press Enter to continue...")
        # Set up KFold
        kf_path = os.path.join(saved_result_path_classification, "kf.pkl")
        if os.path.exists(kf_path):
            with open(kf_path, "rb") as f:
                kf = pickle.load(f)
        else:
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        test_indices = X_df[X_df["subjid"].isin(test_subjects_ids)].index
        train_indices = X_df.index.difference(test_indices)

        if len(test_indices) + len(train_indices) != len(X_df):
            raise ValueError("Mismatch in expected number of indices.")

        train_subjects, test_subjects = (
            X_df.loc[train_indices, "subjid"],
            X_df.loc[test_indices, "subjid"],
        )

        X_df = X_df.drop(columns=["subjid"])
        # print columns
        print("X_df columns: ", X_df.columns)
        # drop gendna_type
        # X_df = X_df.drop(columns=["gendna_type"])

        # print the y distribution
        print("Y distribution:")
        print(y.value_counts())

        X_train_df, X_test_df = X_df.loc[train_indices], X_df.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]

        # print the y_train distribution
        print("Y_train distribution:")
        print(y_train.value_counts())

        # print dimensions
        print("X_train_df shape: ", X_train_df.shape)
        print("X_test_df shape: ", X_test_df.shape)

        # print the columns
        print("X_train_df columns: ", X_train_df.columns)

        features = X_df.columns  # X_df.drop(columns=["subjid"]).columns
        scorer = make_scorer(f1_score, average="weighted")

        # Prepare classifier configuration dictionary
        classifier_config = {
            "kf": kf,
            "scorer": scorer,
            "X_test": X_test_df.values,  # X_test_df.drop(columns=["subjid"]).values,
            "y_test": y_test.values,
            "X_train": X_train_df.values,  # ,X_train_df.drop(columns=["subjid"]).values,
            "y_train": y_train.values,
            "num_folds": num_folds,
            "nFeatures": 25,
            "thr": 0.25,
            "train_subjects": train_subjects,  # X_train_df["subjid"].values,
            "test_subjects": test_subjects,  # X_test_df["subjid"].values,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "features": features,
        }

        os.makedirs(saved_result_path_classification, exist_ok=True)
        with open(classifier_config_path, "wb") as f:
            pickle.dump(classifier_config, f)
            print(f"Classifier configuration saved to {classifier_config_path}")
    else:
        with open(classifier_config_path, "rb") as f:
            classifier_config = pickle.load(f)
        print("Classifier configuration loaded successfully.")

    return (
        classifier_config["X_train"],
        classifier_config["X_test"],
        classifier_config["y_train"],
        classifier_config["y_test"],
        classifier_config["train_subjects"],
        classifier_config["test_subjects"],
        classifier_config["features"],
        classifier_config["kf"],
        classifier_config["scorer"],
        classifier_config["thr"],
        classifier_config["nFeatures"],
        classifier_config["num_folds"],
    )


"""

    # Evaluate model on the test set
    y_pred = best_estimator.predict(X_test)

    print("Best Parameters:\n", best_params)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Feature Importances
    importances = None
    feature_importance_data = None
    if hasattr(best_estimator, "feature_importances_"):
        importances = best_estimator.feature_importances_
        indices_all = np.argsort(importances)
        feature_importance_data = {
            "feature_importances": {features[i]: importances[i] for i in range(len(importances))},
            "top_10_features": {features[i]: importances[i] for i in indices_all[-10:]}
        }
        plot_top_feature_importance(importances, features, clf_model.__class__.__name__, results_path)


    #Save the results
    # Saving results
    results_to_save = {
        "name": clf_model.__class__.__name__,
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

    # Path to save the results
    # Make directory path
    model_results_path = os.path.join(results_path, clf_model.__class__.__name__)

    # Create directories if they do not exist
    os.makedirs(model_results_path, exist_ok=True)

    # Path to save the results
    results_file_path = os.path.join(model_results_path, f"clf_results_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.pkl")
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)    
    print(f"Results saved to {results_file_path}")
    
    

    # Use your updated plot_confusion_matrix function
    confusion_matrix_file = f"cm_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}"
    plot_confusion_matrix(y_test, y_pred, confusion_matrix_file)


    # SHAP Values
    if hasattr(best_estimator, "predict_proba"):
        plot_shap_values(best_estimator, X_train, results_path, clf_model.__class__.__name__, feature_selection_option, balancing_technique)

"""


def perform_classification_best(  # it is also saving the best model results
    clf_model,
    param_grid,
    pipeline,
    X_df,
    X_train,
    X_test,
    y_train,
    y_test,
    kf,
    scorer,
    features,
    balancing_technique,
    feature_selection_option,
):
    """
    Perform classification using the specified classifier.

    Parameters:
        clf_model: Classifier model object (e.g., XGBClassifier, SVC, etc.).
        param_grid: Parameter grid for grid search.
        X_train (array-like): Features of the training set.
        X_test (array-like): Features of the test set.
        y_train (array-like): Target variable of the training set.
        y_test (array-like): Target variable of the test set.
        kf (KFold): Cross-validation iterator.
        scorer (object): Scorer for model evaluation.
        features (list): List of feature names.
        saved_result_path_classification (str): Path to save classification results.

    Returns:
        None
    """

    # handle missing values
    # X_train = np.nan_to_num(X_train, nan=988)
    # X_test = np.nan_to_num(X_test, nan=988)

    # print the number of missing values in X_train and X_test
    # print('Number of missing values in X_train:', np.isnan(X_train).sum())
    # print('Number of missing values in X_test:', np.isnan(X_test).sum())

    # Perform grid search for hyperparameter tuning
    if balancing_technique == "mrmr":
        selector = SequentialFeatureSelector(
            clf_model, direction="backward", scoring=scorer, cv=kf, n_jobs=-1
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("selector", selector),
                ("clf", clf_model),
            ]
        )

        # maybe for RF
        new_keys = {
            "n_estimators": "clf__n_estimators",
            "max_depth": "clf__max_depth",
            "min_samples_split": "clf__min_samples_split",
        }
        # for XGB
        new_keys = {
            "max_depth": "clf__max_depth",
            "n_estimators": "clf__n_estimators",
            "learning_rate": "clf__learning_rate",
        }

        # for SVM
        # new_keys = {"C": "clf__C", "gamma": "clf__gamma", "kernel": "clf__kernel"}

        # Rename keys in param_grid
        param_grid = {new_keys.get(k, k): v for k, v in param_grid.items()}
        param_grid.update(
            {
                "selector__n_features_to_select": [
                    i for i in range(1, len(X_train[0]) + 1)
                ]
            }
        )

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=kf,
            scoring=scorer,
            verbose=1,
            n_jobs=-1,
            return_train_score=True,
        )

    else:
        grid_search = GridSearchCV(
            estimator=clf_model,
            param_grid=param_grid,
            cv=kf,
            scoring=scorer,
            verbose=1,
            n_jobs=-1,
            return_train_score=True,
        )

    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator from grid search
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_score_ = grid_search.best_score_

    cv_results = grid_search.cv_results_
    print("CV RESULTS:______________________________________________________")
    print("Best params:", best_params)
    print("Best estimator:", best_estimator)
    print("Best score:", best_score_)

    # print(cv_results['mean_train_score'])
    # print(cv_results['mean_score_time'])
    # max_mean_train_score = max(cv_results['mean_train_score'])
    # max_mean_test_score = max(cv_results["mean_test_score"])
    # print("Max Mean Train Score: ", max_mean_train_score)
    # print("Max Mean Test Score (F1-weighted): ", max_mean_test_score)
    print("=================================================================")

    # Make predictions on the test set
    y_pred = best_estimator.predict(X_test)

    # Print the best parameters from grid search
    print("Best Parameters:")
    print(grid_search.best_params_)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Define the labels for the confusion matrix
    class_labels = ["nDNA", "mtDNA"]

    # Compute and print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    print("Plotting Confusion Matrix...")
    confusion_matrix_file = f"cm_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}"
    plot_confusion_matrix(
        y_test,
        y_pred,
        os.path.join(saved_result_path_classification, confusion_matrix_file),
    )
    plt.close()

    # Calculate and Plot Importances from xgb methods
    print("Calculating and Plotting Importances...")
    if feature_selection_option == "mrmr":
        try:
            selected_features_mask = best_estimator.named_steps["selector"].support_
            # selected_features_mask = selector.support_
            selected_features = [
                feature
                for feature, selected in zip(features, selected_features_mask)
                if selected
            ]
            print("Selected features:", selected_features)
        except Exception as e:
            print("Error calculating feature importances:", str(e))

    if hasattr(best_estimator, "feature_importances_"):

        importances = best_estimator.feature_importances_

        print("Importances:", importances)
        print("Features:", features)
        feature_importances = {
            features[i]: importances[i] for i in range(len(importances))
        }
        indices_all = np.argsort(importances)  # Sort indices by importance
        feature_importance_data = {
            "feature_importances": feature_importances,
            "top_10_features": {features[i]: importances[i] for i in indices_all},
        }
        print("Feature Importances data:", feature_importance_data)

        # Plot ALL feature importances
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
        feature_importance_file = f"feature_imp_ALL_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.png"
        plt.savefig(
            os.path.join(saved_result_path_classification, feature_importance_file),
            format="png",
            bbox_inches="tight",
        )
        plt.close()

        # Plot ONLY top 10 feature importances
        indices = np.argsort(importances)[
            -10:
        ]  # Select the top 10 most important features
        plt.title("Top 10 Feature Importances", fontsize=15)
        plt.barh(
            range(len(indices)), importances[indices], color="lightblue", align="center"
        )  # Use light blue color
        plt.yticks(
            range(len(indices)), [features[i] for i in indices], ha="right", fontsize=10
        )  # Rotate labels
        plt.xlabel("Relative Importance", fontsize=15)

        feature_importance_file = f"feature_imp_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.png"
        plt.savefig(
            os.path.join(saved_result_path_classification, feature_importance_file),
            format="png",
            bbox_inches="tight",
        )
        plt.close()
    else:
        print("No importances available")

    # Plot SHAP Barh Plot
    if hasattr(clf_model, "predict_proba"):
        try:
            # classifier_model = clf_model.named_steps['clf']  # Extracting the classifier from the pipeline
            explainer = shap.Explainer(best_estimator, X_df)
            # shap_values = explainer(X_test)
            shap_values = explainer(X_df)
            shap.plots.bar(shap_values)

            # Save the plot as a PNG file
            shap_bar_plot_file = f"shap_bar_plot_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.png"
            plt.savefig(
                os.path.join(saved_result_path_classification, shap_bar_plot_file),
                format="png",
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            print("Error plotting SHAP bar plot:", str(e))

    best_estimator_file = f"best_estimator_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.pkl"
    with open(
        os.path.join(saved_result_path_classification, best_estimator_file), "wb"
    ) as f:
        pickle.dump(best_estimator, f)
    print(f"Best estimator saved to {best_estimator_file}")

    ## Save all relevant results to a file
    # Save all relevant results to a file
    results_to_save = {
        "best_params": best_params,
        "best_estimator": best_estimator,
        "best_score": best_score_,
        "cv_results": cv_results,
        "y_pred": y_pred,
        "y_test": y_test,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": conf_matrix,
        "importances": importances,
        "feature_importance_data": feature_importance_data,
    }

    # Save the best estimator to a file
    best_estimator_file = f"best_estimator_{clf_model.__class__.__name__}_{feature_selection_option}_{balancing_technique}.pkl"
    with open(
        os.path.join(saved_result_path_classification, best_estimator_file), "wb"
    ) as f:
        pickle.dump(best_estimator, f)
    print(f"Best estimator saved to {best_estimator_file}")


from sklearn.impute import SimpleImputer


def balance_data(X_train, y_train, X_test, y_test, balancing_technique):
    """
    Balance the data using the specified balancing technique.

    Parameters:
        X_train (array-like): Features of the training set.
        y_train (array-like): Target variable of the training set.
        X_test (array-like): Features of the test set.
        y_test (array-like): Target variable of the test set.
        balancing_technique (str): Technique to use for balancing.
            Options: 'under', 'over', 'unders_test', 'over_test', 'none'.

    Returns:
        X_train_resampled (array-like): Resampled features of the training set.
        y_train_resampled (array-like): Resampled target variable of the training set.
        X_test_resampled (array-like): Resampled features of the test set.
        y_test_resampled (array-like): Resampled target variable of the test set.
    """
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")  # Use mean imputation
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    if balancing_technique == "under":
        # Apply undersampling to training data
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = (
            X_test_imputed,
            y_test,
        )  # No resampling for test data
    elif balancing_technique == "smote":
        # Apply oversampling to training data
        oversampler = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = (
            X_test_imputed,
            y_test,
        )  # No resampling for test data
    elif balancing_technique == "ada":
        # Apply oversampling to training data
        oversampler = ADASYN(random_state=42)  # TO DO update
        X_train_resampled, y_train_resampled = oversampler.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = (
            X_test_imputed,
            y_test,
        )  # No resampling for test data

    elif balancing_technique == "unders_test":
        # Apply undersampling to both training and test data
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = undersampler.fit_resample(
            X_test_imputed, y_test
        )
    elif balancing_technique == "over_test":
        # Apply oversampling to training data and undersampling to test data
        oversampler = SMOTE(random_state=42)
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = undersampler.fit_resample(
            X_test_imputed, y_test
        )
    elif balancing_technique == "smoteenn":
        print("Applying SMOTEENN (SMOTE + ENN) for data balancing...")
        balancer = (
            SMOTEENN()
        )  # Edited Nearest Neighbors (ENN) is used for cleaning the data
        X_train_resampled, y_train_resampled = balancer.fit_resample(
            X_train_imputed, y_train
        )
        X_test_resampled, y_test_resampled = X_test_imputed, y_test

    elif balancing_technique == "balanced_random_forest":
        print("Applying Balanced Random Forest for data balancing...")
        balancer = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
        balancer.fit(X_train_imputed, y_train)  # Fit the model
        X_train_resampled, y_train_resampled = (
            balancer.predict(X_train_imputed),
            y_train,
        )
        X_test_resampled, y_test_resampled = balancer.predict(X_test_imputed), y_test

    else:
        # No balancing technique applied
        return X_train_imputed, y_train, X_test_imputed, y_test

    # Print the class distribution after resampling
    print("Balancing Technique:", balancing_technique)
    print(
        "Training Set Class Distribution after Resampling:", Counter(y_train_resampled)
    )
    print("Test Set Class Distribution after Resampling:", Counter(y_test_resampled))

    return X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled


def define_X_y(df):
    """Select and drop unnecessary features from the DataFrame."""
    df_vars = pd.read_excel(important_vars_path)
    column_name = "consider for mtDNA vs nDNA classification?"
    columns_to_drop = list(df_vars.loc[df_vars[column_name] == "N", "variable"])

    # Additional columns to drop
    additional_columns = [
        "Hospital",
        # "nDNA",
        # "mtDNA",
        "gendna_type",
        "gendna_type_num",
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

    y = df["gendna_type"]
    df = df.drop(columns=["gendna_type", "test", "Unnamed: 0"])
    X = df.values

    return X, y


def print_data_info(X_train, X_test, y_train, y_test, features, df, working_path):
    """Prints information about the datasets and features."""
    # print the experiment path
    print("Working path:", working_path)

    print("Dataset for classification shape:", df.shape)

    print("Dimension of X_train:", X_train.shape)
    print("Dimension of X_test:", X_test.shape)
    print("Dimension of y_train:", y_train.shape)
    print("Dimension of y_test:", y_test.shape)
    print("Features names:", features)

    # print distribution of y_train and y_test
    print("Distribution of y_train:", Counter(y_train))
    print("Distribution of y_test:", Counter(y_test))


def fill_missing_values(df):
    # Load the important variables file
    df_vars = pd.read_excel(important_vars_path)

    # Print columns not in the important variables list
    missing_columns = [
        col for col in df.columns if col not in df_vars["variable"].values
    ]
    # Fill missing values based on the important variables list
    for index, row in df_vars.iterrows():
        column = row["variable"]
        na = row["NA"]
        if column in df.columns:
            if na == "998":
                df[column].fillna(998, inplace=True)
            # else:
            # df[column].fillna(998, inplace=True)
    return df


def load_and_prepare_data(GLOBAL_DF_PATH, EXPERIMENT_PATH):
    """Load and prepare the DataFrame for classification."""
    print("Global DataFrame path:", GLOBAL_DF_PATH)

    if not os.path.exists(GLOBAL_DF_PATH):
        raise FileNotFoundError(f"File not found: {GLOBAL_DF_PATH}")

    df = pd.read_csv(GLOBAL_DF_PATH)

    # Display initial DataFrame information
    nRow, nCol = df.shape
    print(f'The DataFrame "df_preprocessed" contains {nRow} rows and {nCol} columns.')

    plot_gendna_distribution(df, EXPERIMENT_PATH)
    # print that distribution plot is saved
    print(f"gendna distribution plot is saved in {EXPERIMENT_PATH}")

    mt_DNA_patients = df[df["gendna_type"] == 0]

    return df, mt_DNA_patients


def process_feature_selection(
    clf_model,
    X_df,
    X_train,
    X_test,
    y_train,
    param_grid,
    pipeline,
    scorer,
    kf,
    feature_selection,
    features=None,
    num_folds=None,
    nFeatures=None,
    thr=None,
):
    """
    Process feature selection based on the specified technique.

    Parameters:
    - X_train (numpy array): Training feature matrix.
    - X_test (numpy array): Test feature matrix.
    - y_train (numpy array): Target labels for the training data.
    - feature_selection (str): The feature selection technique to apply ('no', 'mrmr', 'pca').
    - features (list): List of feature names (required for 'mrmr' feature selection).
    - num_folds (int): Number of folds for cross-validation in 'mrmr' feature selection.
    - nFeatures (int): Number of top features to select in 'mrmr' feature selection.
    - thr (float): Threshold for feature selection in 'mrmr' feature selection.

    Returns:
    - X_train_selected (numpy array): Selected features for training data.
    - X_test_selected (numpy array): Selected features for test data.
    """

    param_grid_selected = param_grid
    pipeline_selected = pipeline

    # Remove missing values

    if feature_selection == "mrmr_ff":
        #TODO implement here meme
        print("Feature ranking using MRMR_FF")
    elif feature_selection == "pca":
        print("Starting PCA")
        imputer_X = SimpleImputer(strategy="mean")
        X_train = imputer_X.fit_transform(X_train)
        X_test = imputer_X.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_selected = pca.fit_transform(X_train_scaled)
        X_test_selected = pca.transform(X_test_scaled)

        # Get the loading vectors (components)
        components = pca.components_

        # Create a DataFrame to display the components
        feature_names = features  # Original feature names
        components_df = pd.DataFrame(components, columns=feature_names)

        # print("PCA Components (Loading Vectors):")
        # print(components_df)
        # Save PCA components to a file
        path_csv_pca = os.path.join(
            saved_result_path_classification, "pca_components.csv"
        )
        components_df.to_csv(path_csv_pca, index=False)
        print("PCA components saved to pca_components.csv")

        # If you want to see which features contribute most to each principal component
        zero_contribution_features = []
        for i, component in enumerate(components_df.values):
            sorted_indices = component.argsort()[::-1]
            sorted_feature_names = feature_names[sorted_indices]
            sorted_component_values = component[sorted_indices]
            # print(f"\nPrincipal Component {i + 1}:")
            for feature_name, component_value in zip(
                sorted_feature_names, sorted_component_values
            ):
                # print(f"{feature_name}: {component_value}")
                if component_value == 0:
                    zero_contribution_features.append(feature_name)
        # Print features with component_value equal to zero (without duplicates)
        # check if not empty:
        if zero_contribution_features:
            print("\nFeatures with component value equal to zero:")
            zero_contribution_features = list(set(zero_contribution_features))
            print(zero_contribution_features)

    elif feature_selection == "mrmr":
        print("Feature ranking using MRMR")
        top_feature_indices_mrmr = rankfeatures(
            X_train, y_train, "mrmr", num_folds, nFeatures, thr, kf
        )
        selectedFeatures_mrmr = []
        for subset_size in range(
            len(top_feature_indices_mrmr) - 1, len(top_feature_indices_mrmr) + 1
        ):
            feature_combinations_mrmr = list(
                combinations(top_feature_indices_mrmr, subset_size)
            )
            selectedFeatures_mrmr.extend(feature_combinations_mrmr)

        print("Top " + str(thr * 100) + "% Feature Indices:", top_feature_indices_mrmr)
        selectedFeatures = selectedFeatures_mrmr
        selectedFeatures_name = [features[i] for i in selectedFeatures[-1]]
        print("selected Features names", selectedFeatures_name)

        # Create subsets of features for the final model
        selectedFeatures_set = selectedFeatures[-1]
        X_train_selected = X_train[:, selectedFeatures_set]
        X_test_selected = X_test[:, selectedFeatures_set]
        print("Selected Features of Subsets:", selectedFeatures_set)

        # Check if the model is tree-based, which does not require SequentialFeatureSelector
        if isinstance(clf_model, (DecisionTreeClassifier, RandomForestClassifier, SVC)):
            # Tree-based models often do not require explicit feature selection
            pipeline_selected = Pipeline(
                [("imputer", SimpleImputer(strategy="mean")), ("clf", clf_model)]
            )
        else:
            # Use SequentialFeatureSelector for non-tree-based models
            selector = SequentialFeatureSelector(
                clf_model, direction="backward", scoring=scorer, cv=kf, n_jobs=-1
            )

            param_grid_selected = {
                **param_grid,
                "selector__n_features_to_select": range(
                    1, len(X_train_selected[0]) + 1
                ),
            }

            pipeline_selected = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("selector", selector),
                    ("clf", clf_model),
                ]
            )
    elif feature_selection == "select_from_model":
        print("Feature selection using SelectFromModel...")
        try:
            selector = SelectFromModel(estimator=clf_model)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Rimuove la colonna 'subjid' se esiste in X_df
            if "subjid" in X_df.columns:
                X_df = X_df.drop(columns=["subjid"])

            selected_features = X_df.columns[selector.get_support()]
            print("Selected Features:")
            print(selected_features)
            print("Number of Selected Features:", len(selected_features))

        except Exception as e:
            print(f"Error during feature selection: {e}")
            print(
                "Falling back to default configuration (no feature selection applied)."
            )

            # Nessuna selezione di feature applicata
            X_train_selected = X_train
            X_test_selected = X_test
            param_grid_selected = param_grid
            pipeline_selected = pipeline

    elif feature_selection == "rfe":
        print("Feature selection using Recursive Feature Elimination (RFE)...")
        try:
            selector = RFE(
                estimator=clf_model, n_features_to_select=nFeatures, step=5
            )  # step to reduce the number of features

            X_train_selected = selector.fit_transform(X_train, y_train)

            X_test_selected = selector.transform(X_test)
            # Remove the column subjid
            X_df = X_df.drop(columns=["subjid"])
            selected_features = X_df.columns[selector.get_support()]
            print("Selected Features:")
            print(selected_features)
            print("Number of Selected Features:", len(selected_features))
        except Exception as e:
            print(f"Error during feature selection: {e}")
            print(
                "Falling back to default configuration (no feature selection applied)."
            )

            # Nessuna selezione di feature applicata
            X_train_selected = X_train
            X_test_selected = X_test
            param_grid_selected = param_grid
            pipeline_selected = pipeline

    elif feature_selection == "mrmr_selected":
        selected_features = [
            "ear_voice_abn",
            "genit_breast_abn",
            "symp_on_2",
            "eye_abn",
            "card_abn",
            "internal_abn",
            "alcohol",
            "ecgh",
            "resp",
            "hba1c",
            "sex",
            "ear",
            "lac",
            "echo",
        ]
        X_df = X_df[selected_features]
        X_df.reset_index(drop=True, inplace=True)

        # Check if X_train is a numpy array and convert it to DataFrame if necessary
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_train.reset_index(drop=True, inplace=True)

        # Select rows based on the index and convert it back to numpy array
        X_train_selected = X_df.loc[X_train.index].values

        # Do the same for X_test
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
            X_test.reset_index(drop=True, inplace=True)

        X_test_selected = X_df.loc[X_test.index].values

        param_grid_selected = param_grid
        pipeline_selected = pipeline
    elif (
        feature_selection == "mrmr_selected_all"
    ):  # mrmr with 25% selected features on the dataset with all patients, even without symp
        selected_features = [
            "ear_voice_abn",
            "genit_breast_abn",
            "lac",
            "card_abn",
            "resp",
            "internal_abn",
            "ecgh",
            "bmi",
            "ear",
            "sex",
            "hba1c",
            "eye_abn",
            "ecg",
            "hmri",
        ]
        X_df = X_df[selected_features]
        X_df.reset_index(drop=True, inplace=True)

        # Check if X_train is a numpy array and convert it to DataFrame if necessary
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_train.reset_index(drop=True, inplace=True)

        # Select rows based on the index and convert it back to numpy array
        X_train_selected = X_df.loc[X_train.index].values

        # Do the same for X_test
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
            X_test.reset_index(drop=True, inplace=True)

        X_test_selected = X_df.loc[X_test.index].values

        param_grid_selected = param_grid
        pipeline_selected = pipeline
    else:
        # No feature selection applied

        X_train_selected = X_train
        X_test_selected = X_test
        param_grid_selected = param_grid
        pipeline_selected = pipeline

    return X_train_selected, X_test_selected, param_grid_selected, pipeline_selected
