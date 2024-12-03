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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

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
)
from utilities import *
from processing import *
from plotting import *


def classification_results_80_20(X, y, classifier_name, clf, param_grid):
    prediction = []
    gt = []
    proba = []

    scaler = StandardScaler()
    # fold = 0
    f1_scores = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )  # 20%test size

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Use the best estimator from grid search
    clf = grid_search.best_estimator_
    importances = grid_search.best_estimator_.feature_importances_

    # Fit the model and make predictions
    clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    pr = clf.predict_proba(X_test)

    prediction.append(res)  # save
    gt.append(y_test)  # calc cm su y_test e pr
    proba.append(pr)

    # confusion matrix for each fold!
    conf_mat = metrics.confusion_matrix(y_test, res)  # y_test, X test and SAVE IT
    confusion_matrix_plot(
        conf_mat,
        f"Confusion Matrix - {classifier_name} Classification - ",
        f"CM_{classifier_name}_",
    )  # Confusion matrix and name to save  confusion matrix
    plt.close()

    best_pars = grid_search.best_params_
    print(best_pars)

    fold_f1_score = f1_score(y_test, res)
    f1_scores.append(fold_f1_score)  # Store f1_score for this fold

    gt = np.hstack(gt)
    prediction = np.hstack(prediction)
    pp = np.vstack(proba)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    results = {
        "test_accuracy": [accuracy_score(y_test, res)],
        "test_precision": [precision_score(y_test, res)],
        "test_recall": [recall_score(y_test, res)],
        "test_f1_score": [f1_score(y_test, res)],
    }

    # save_variable(results, name + '_results')
    # save_variable([gt, prediction, pp, X_train, X_test, y_train, y_test, clf, results, f1_scores], name)  # Save variables
    clf_performances(results, f1_scores, classifier_name)  # Print performances
    return [
        gt,
        prediction,
        pp,
        X_train,
        X_test,
        y_train,
        y_test,
        grid_search,
        results,
        f1_scores,
        importances,
    ]


def classification_results_10cv(X, y, name, clf, param_grid):
    prediction = []
    gt = []
    proba = []

    scaler = StandardScaler()
    fold = 0
    f1_scores = []

    # Perform 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        fold += 1
        print(f"Fold: {fold}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Perform grid search on each fold
        grid_search = GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Use the best estimator from grid search
        clf = grid_search.best_estimator_

        # Fit the model and make predictions
        clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        pr = clf.predict_proba(X_test)

        prediction.append(res)
        gt.append(y_test)
        proba.append(pr)

        # Confusion matrix for each fold
        conf_mat = metrics.confusion_matrix(y_test, res)
        confusion_matrix_plot(
            conf_mat,
            f"Confusion Matrix - {name} Classification - Fold: {fold}",
            f"CM_{name}_fold_{fold}",
        )
        plt.close()

        best_pars = grid_search.best_params_
        print(best_pars)

        fold_f1_score = f1_score(y_test, res)
        f1_scores.append(fold_f1_score)

    gt = np.hstack(gt)
    prediction = np.hstack(prediction)
    pp = np.vstack(proba)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    # Use the best estimator from the last fold for overall evaluation
    overall_clf = grid_search.best_estimator_

    results = model_selection.cross_validate(
        estimator=overall_clf, X=X, y=y, cv=skf, scoring=scoring
    )

    clf_performances(results, f1_scores, name)
    return [
        gt,
        prediction,
        pp,
        X_train,
        X_test,
        y_train,
        y_test,
        overall_clf,
        results,
        f1_scores,
    ]


def clf_performances(results, f1_scores, name):
    median_scores = {
        "Accuracy": np.median(results["test_accuracy"]),
        "Precision": np.median(results["test_precision"]),
        "F1 Score": np.median(f1_scores),
    }

    q1_scores = {
        "Accuracy": np.percentile(results["test_accuracy"], 25),
        "Precision": np.percentile(results["test_precision"], 25),
        "F1 Score": np.percentile(f1_scores, 25),
    }

    q3_scores = {
        "Accuracy": np.percentile(results["test_accuracy"], 75),
        "Precision": np.percentile(results["test_precision"], 75),
        "F1 Score": np.percentile(f1_scores, 75),
    }

    print("Median Scores:")
    print(median_scores)
    print("1st Quartile Scores:")
    print(q1_scores)
    print("3rd Quartile Scores:")
    print(q3_scores)

    # Save median, 1st quartile, and 3rd quartile of f1_score
    f1_scores_data = {
        "Median": np.median(f1_scores),
        "1st Quartile": np.percentile(f1_scores, 25),
        "3rd Quartile": np.percentile(f1_scores, 75),
    }
    save_variable(f1_scores_data, name + "_f1_2080", saved_result_path_classification)


def classification_results(X, y, name, clf, skf):
    prediction = []
    gt = []
    proba = []

    scaler = StandardScaler()
    fold = 0
    f1_scores = []

    # clf = GridSearchCV(estimator=clf_model, param_grid=params, cv=skf, scoring='f1', n_jobs=-1)

    idx = skf.split(X, y)  # Constant splitting

    for train_index, test_index in idx:
        fold = fold + 1
        print("Fold:")
        print(fold)

        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index].tolist(), X[test_index].tolist()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        pr = clf.predict_proba(X_test)

        prediction.append(res)  # save
        gt.append(y_test)  # calc cm su y_test e pr
        proba.append(pr)

        best_pars = clf.best_params_
        print(best_pars)

        fold_f1_score = f1_score(y_test, res)
        f1_scores.append(fold_f1_score)  # Store f1_score for this fold

    gt = np.hstack(gt)
    prediction = np.hstack(prediction)
    pp = np.vstack(proba)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    results = model_selection.cross_validate(
        estimator=clf, X=X, y=y, cv=skf, scoring=scoring
    )

    # save_variable(results, name + '_results')
    # save_variable([gt, prediction, pp, X_train, X_test, y_train, y_test, clf, results, f1_scores], name)  # Save variables
    clf_performances(results, f1_scores, name)  # Print performances
    return [
        gt,
        prediction,
        pp,
        X_train,
        X_test,
        y_train,
        y_test,
        clf,
        results,
        f1_scores,
    ]

    # plt.close()

    # scores = [f1_scores_data['Median'], f1_scores_data['1st Quartile'], f1_scores_data['3rd Quartile']]

    # plt.figure(figsize=(6, 4))
    # plt.boxplot(scores, vert=False)
    # plt.title('F1 Score Distribution')
    # plt.xlabel('Scores')
    # plt.yticks([1], ['F1 Score'])
    # plt.show()
    # global_path = '/content/drive/MyDrive/Colab Notebooks/GENOMIT'
    # saved_result_path_classification = os.path.join(global_path, 'saved_results')
    # plt.savefig(os.path.join(saved_result_path_classification, (file_name+'.png')),format="png", bbox_inches="tight")


def perform_classifier_operations(classifier_name, columns, target_names):
    # Load class results
    [
        gt,
        prediction,
        pp,
        X_train,
        X_test,
        y_train,
        y_test,
        clf,
        results,
        f1_scores,
        importances,
    ] = load_pickle_variable(
        classifier_name + "_results2080", saved_result_path_classification
    )

    # Print the results
    # Best score achieved during the GridSearchCV
    print(
        "GridSearch CV best score for {}: {:.4f}\n".format(
            classifier_name, clf.best_score_
        )
    )
    # Print parameters that give the best results
    print(
        "Parameters that give the best results for {}:\n".format(classifier_name),
        clf.best_params_,
    )
    # Print estimator that was chosen by the GridSearch
    print(
        "\nEstimator that was chosen by the search for {}:\n".format(classifier_name),
        clf.best_estimator_,
    )
    target_names = ["Childhood", "Adults"]
    print(classification_report(gt, prediction, target_names=target_names))
    clf_performances(results, f1_scores, classifier_name)

    # Plot Precision-recall curve
    precision_recall_plot(
        gt,
        pp,
        "Precision-Recall Curve - {} Classification - ".format(classifier_name),
        f"PR_{classifier_name}_2080",
        global_path,
    )
    plt.close()

    # Confusion matrix
    conf_mat = metrics.confusion_matrix(gt, prediction)
    # Plot confusion matrix
    confusion_matrix_plot(
        conf_mat,
        "Confusion Matrix - {} Classification - ".format(classifier_name),
        f"CM_{classifier_name}_2080.png",
    )  # Confusion matrix and name to save  confusion matrix
    plt.close()
    # Get feature importances from the best estimator
    # importances = clf.best_estimator_.feature_importances_

    # Call the plot_top_feature_importance function to plot and save the top 10 feature importance plot
    print("importances: ", len(importances))
    print("columns:", len(columns))

    plot_top_feature_importance(
        importances, columns, f"FI_{classifier_name}_2080", global_path, top_n=10
    )
    plt.close()
    # Load F1 scores data
    f1_scores_data = load_pickle_variable(
        classifier_name + "_f1_2080", saved_result_path_classification
    )
    # plot_f1_scores_boxplot(f1_scores_data, 'BX_{}_'.format(classifier_name),saved_result_path_classification)
    plt.close()


def calculate_percentages():
    # Given data
    nerv_involved = 112
    nerv_not_involved = 124

    normal = 164
    nDNA_mtDNA = 616

    mono_oligo = 544
    multi = 236

    aao_60_minus = 291
    aao_60_plus = 409

    # Calculate percentages
    nerv_percentage = {
        "nerv_involved": nerv_involved / (nerv_involved + nerv_not_involved) * 100,
        "nerv_not_involved": nerv_not_involved
        / (nerv_involved + nerv_not_involved)
        * 100,
    }

    mtDNA_percentage = {
        "normal": normal / (normal + nDNA_mtDNA) * 100,
        "nDNA_mtDNA": nDNA_mtDNA / (normal + nDNA_mtDNA) * 100,
    }

    mono_multi_percentage = {
        "mono_oligo": mono_oligo / (mono_oligo + multi) * 100,
        "multi": multi / (mono_oligo + multi) * 100,
    }

    aao_percentage = {
        "60_minus": aao_60_minus / (aao_60_minus + aao_60_plus) * 100,
        "60_plus": aao_60_plus / (aao_60_minus + aao_60_plus) * 100,
    }

    return nerv_percentage, mtDNA_percentage, mono_multi_percentage, aao_percentage


# %% Feature ranking on the training set using cross-validation
def rankfeatures(X_train, Y_train, frmethod, nFolds, nFeatures, thr, kf):

    Y_train = np.asarray(Y_train)

    # print('Y_train', Y_train)
    top_features_all = []

    if frmethod == "relieff":
        relieff = ReliefF(n_neighbors=5)

    # Iterate over training sets, leaving out data from N/5 patients at a time
    patient_indices = np.arange(X_train.shape[0])  # np.unique(S)
    # print('patient_indices', patient_indices)
    # Shuffle the list of patients
    np.random.seed(42)
    np.random.shuffle(patient_indices)
    # Use KFold to split the shuffled indices into folds
    # kf = KFold(n_splits=nFolds)

    # Iterate over the folds
    for fold, (train_indices, val_indices) in enumerate(kf.split(patient_indices)):

        X_train_cv = X_train[train_indices]
        y_train_cv = Y_train[train_indices]

        # print('X_train_cv shape: ', X_train_cv.shape)
        # print('y_train_cv shape: ', y_train_cv.shape)

        if frmethod == "relieff":
            print(f"Relieff feature ranking - Fold {fold+1} of {nFolds}")
            # Feature ranking using ReliefF
            relieff.fit(X_train_cv, y_train_cv)
            # Get the weights of the features
            feature_weights = relieff.feature_importances_
            top_feature_indices = np.argsort(feature_weights)[::-1][:nFeatures]
        elif frmethod == "mrmr":
            print(f"MRMR feature ranking - Fold {fold+1} of {nFolds}")
            # Feature ranking using MRMR
            top_feature_indices = mrmr.mrmr_regression(
                X=pd.DataFrame(X_train),
                y=pd.DataFrame(Y_train),
                K=int(thr * X_train.shape[1]),
            )

        top_features_all.append(top_feature_indices)

    top_features_all_ = np.asarray(top_features_all, dtype=np.float32)
    # print('Found top_features_all_',top_features_all_)

    # Find all best features across the folds
    all_top_features = np.unique(top_features_all_)

    # For each best feature, retrieve the corresponding rank in each fold
    feature_scores = np.empty([len(all_top_features), 2], dtype=int)
    ii = 0
    for ff in all_top_features:
        idx_ff = np.asarray(np.where(np.isin(top_features_all_, ff)))
        score = np.sum(nFeatures - idx_ff[1])
        feature_scores[ii, :] = [ff, score]
        ii += 1

    sorted_scores = np.argsort(feature_scores[:, 1])[::-1]
    # top_features = feature_scores[sorted_scores[:4], 0]
    top_features = feature_scores[sorted_scores[: int(thr * X_train.shape[1])], 0]
    # top_features = feature_scores[sorted_scores[:nFeatures], 0]

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


def process_gendna_column(df):
    """
    Process the 'gendna' column in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the 'gendna' column.
        saved_result_path_classification (str): The path to save the result files.

    Returns:
        DataFrame: The DataFrame with processed 'gendna' column.
    """
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
        saved_result_path_classification, "patients_with_nan_gendna.txt"
    )
    patients_with_nan_gendna = df[df["gendna"].isnull()][["subjid", "Hospital"]]
    patients_with_nan_gendna.to_csv(file_path, index=False)
    print("Patients with NaN 'gendna' information saved to:", file_path)

    # Drop NaN values from 'gendna' column
    df_non_nan = df.dropna(subset=["gendna"])

    # Remove rows where 'gendna' is equal to 1
    df_non_nan = df_non_nan[df_non_nan["gendna"] != 1]

    # Create 'nDNA' and 'mtDNA' classes
    df_non_nan["nDNA"] = df_non_nan["gendna"].apply(
        lambda x: "nDNA" if x in [4, 6, 8] else None
    )
    df_non_nan["mtDNA"] = df_non_nan["gendna"].apply(
        lambda x: "mtDNA" if x in [5, 7] else None
    )

    # Combine 'nDNA' and 'mtDNA' classes into 'gendna_type' column
    df_non_nan["gendna_type"] = df_non_nan.apply(
        lambda row: row["nDNA"] if row["nDNA"] is not None else row["mtDNA"], axis=1
    )

    # Print distribution of 'gendna_type'
    # print("\nDistribution of 'gendna_type':")
    # print(df_non_nan['gendna_type'].value_counts())

    # Convert 'gendna_type' into numerical values
    df_non_nan["gendna_type"] = df_non_nan["gendna_type"].replace(
        {"mtDNA": 0, "nDNA": 1}
    )
    plot_gendna_distribution(df_non_nan)

    # Convert DataFrame to numerical format
    df_processed = convert_to_numerical(df_non_nan)

    print("\nDistribution of 'gendna_type' in the numerical DataFrame:")
    print(df_processed["gendna_type"].value_counts())

    # Plot distribution of 'gendna_type'

    print("Distribution of 'gendna_type' plotted.")

    return df_processed, df_non_nan


def define_X_y(df, columns_to_drop):
    """
    Define X and y from the given DataFrame, dropping specified columns.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        global_path (str): The path to the global directory.
        columns_to_drop (list): List of column names to drop from the DataFrame.

    Returns:
        tuple: A tuple containing X (features) and y (target).
    """
    # Extract the target variable
    y = df["gendna_type"]
    # y = df.pop("gendna_type")

    # Load the important variables from Excel
    important_vars_path = os.path.join(
        global_path, "dataset", "important_variables.xlsx"
    )
    df_vars = pd.read_excel(important_vars_path)

    # Specify the column name for considering variables
    column_name = "consider for mtDNA vs nDNA classification?"
    print(df.columns)

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

    return X, y, X_df


def fill_missing_values(df):
    # Load the important variables file
    important_vars_path = os.path.join(
        global_path, "dataset", "important_variables.xlsx"
    )
    df_vars = pd.read_excel(important_vars_path)

    # Print columns not in the important variables list
    missing_columns = [
        col for col in df.columns if col not in df_vars["variable"].values
    ]
    # if missing_columns:
    # print("Columns not in the important variables list:")
    # for col in missing_columns:
    # print(col)

    # Fill missing values based on the important variables list
    for index, row in df_vars.iterrows():
        column = row["variable"]
        na = row["NA"]
        if column in df.columns:
            if na == "NA":
                df[column].fillna(-1, inplace=True)
            # else:
            # df[column].fillna(998, inplace=True)

    # Print the updated missing value counts
    # print("Updated missing value counts:")
    for column in df.columns:
        if column in df_vars["variable"].values:
            # print(f"{column}: {df[column].isnull().sum()}")
            continue

    return df


def experiment_definition(X, y, X_df, num_folds=5):
    """
    Define experiment parameters and split the data into train and test sets.

    Parameters:
        X (array-like): Features.
        y (array-like): Target variable.
        num_folds (int): Number of folds for cross-validation.
        nFeatures (int): Maximum number of features to select.
        thr (float): Threshold to select top features based on importance.

    Returns:
        tuple: Train and test sets along with cross-validation iterator, scorer.
    """
    np.random.seed(42)
    # check if the kf is already saved
    if os.path.exists(os.path.join(saved_result_path_classification, "kf.pkl")):
        kf = pd.read_pickle(os.path.join(saved_result_path_classification, "kf.pkl"))
        print("kf", kf)
    else:
        # Use KFold to split the shuffled indices into folds
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        with open(os.path.join(saved_result_path_classification, "kf.pkl"), "wb") as f:
            pickle.dump(kf, f)

    # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Split the data into training and test sets (80-20 split)
    print("Splitting started")

    subjid = X_df["subjid"]
    # print("subjid", subjid)
    # Resetting the index to ensure it's a fresh, clean index
    X_df = X_df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
    # X_test_df = pd.DataFrame(X_test, columns=X_df.columns)

    # Align y_test with X_test_df
    y_test = y_test.reset_index(drop=True)

    # Convert split arrays back to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=X_df.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test, columns=X_df.columns, index=X_test.index)

    train_subjects = X_train_df["subjid"]
    test_subjects = X_test_df["subjid"]

    # Get the indices for the train and test sets
    train_indices = X_train_df.index
    test_indices = X_test_df.index

    print("train_indices", train_indices)
    print("len train indeces", len(train_indices))
    print("test_indices", test_indices)

    # Verification
    consistent_order = (test_subjects.index == y_test.index).all()
    print(
        f"Is the order of 'subjid' in X_test_df the same as 'y_test'? {consistent_order}"
    )
    print("Test Subjects in X_test_df:")
    print(test_subjects.values)

    print("Corresponding y_test values:")
    print(y_test.values)

    # print len of train and test subjects and their values
    print("len of train subjects", len(train_subjects))
    print("len of test subjects", len(test_subjects))
    # Print the subjid to check which patients are in the training and test sets
    print("Training set subjects:\n", train_subjects)
    print("Test set subjects:\n", test_subjects)

    # print len unique train and test subjects
    print("LEn Unique test subjects", len(test_subjects.unique()))
    print("LEn Unique train subjects", len(train_subjects.unique()))

    # time.pause(70)

    # Drop the subjid column from X_train and X_test if no longer needed
    X_train = X_train_df.drop(columns=["subjid"]).values
    X_test = X_test_df.drop(columns=["subjid"]).values

    # check the type of X_train and X_test
    print("Type of X_train", type(X_train))
    print("Type of X_test", type(X_test))

    # print size of X_train and X_test that are arrays
    print("X_train size", X_train.shape)
    print("X_test size", X_test.shape)

    print("Splitting completed")

    # Create a scorer for F1 score
    scorer = make_scorer(f1_score, average="weighted")

    nFeatures = 15  # Maximum number of features to select
    thr = 0.25  # threshold to select top 25% features
    num_folds = 5  # for mrmr

    # Save classifier configuration to a file
    config_file = saved_result_path_classification + "/classifier_configuration.pkl"
    classifier_config = {
        "kf": kf,
        "scorer": scorer,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "num_folds": num_folds,
        "nFeatures": nFeatures,
        "thr": thr,
        "num_folds": num_folds,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    # with open(config_file, "wb") as f:
    # pickle.dump(classifier_config, f)
    # print(f"Classifier configuration saved to {config_file}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_subjects,
        test_subjects,
        kf,
        scorer,
        thr,
        nFeatures,
        num_folds,
    )


def perform_classification(
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
    if balancing_technique == "mrmr":  # TODO: Update
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
    if balancing_technique == "mrmr":  # TODO: Update
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

    # Save the best estimator to a file
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
    # Path to save the results
    results_file_path = os.path.join(
        saved_result_path_classification, "classification_results.pkl"
    )
    with open(results_file_path, "wb") as f:
        pickle.dump(results_to_save, f)


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
    elif balancing_technique == "over":
        # Apply oversampling to training data
        oversampler = SMOTE(random_state=42)
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

    if feature_selection == "pca":
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

        # create subsets of features for BACKWARD RANKING
        selectedFeatures_set = selectedFeatures[-1]
        X_train_selected = X_train[:, selectedFeatures_set]
        X_test_selected = X_test[:, selectedFeatures_set]

        print("Selected Features of Subsets:", selectedFeatures_set)

        selector = SequentialFeatureSelector(
            clf_model, direction="backward", scoring=scorer, cv=kf, n_jobs=-1
        )

        # update parameter grid:
        param_grid_selected = {
            **param_grid,
            "selector__n_features_to_select": range(1, len(X_train_selected[0]) + 1),
        }  # Limit the number of features to select}
        pipeline_selected = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("selector", selector),
                ("clf", clf_model),
            ]
        )
    elif feature_selection == "select_from_model":
        print("Feature selection using SelectFromModel...")
        selector = SelectFromModel(estimator=clf_model)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        selected_features = X_df.columns[selector.get_support()]
        print("Selected Features:")
        print(selected_features)
        print("Number of Selected Features:", len(selected_features))

    elif feature_selection == "rfe":
        print("Feature selection using Recursive Feature Elimination (RFE)...")
        selector = RFE(estimator=clf_model, n_features_to_select=nFeatures)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_features = X_df.columns[selector.get_support()]
        print("Selected Features:")
        print(selected_features)
        print("Number of Selected Features:", len(selected_features))

    else:
        # No feature selection applied
        X_train_selected = X_train
        X_test_selected = X_test
        param_grid_selected = param_grid
        pipeline_selected = pipeline

    return X_train_selected, X_test_selected, param_grid_selected, pipeline_selected


# %%
