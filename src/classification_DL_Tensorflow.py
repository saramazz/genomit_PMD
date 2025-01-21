# MLP and NN
"""
Code to do the classification using MLP
"""
# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import sys
from datetime import datetime


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Local imports
from config import (
    global_path,
    saved_result_path_classification,
    saved_result_path_classification_models,
)
from utilities import *
from preprocessing import *
from processing import *
from plotting import *

# Standard library imports
import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set a random seed for reproducibility
np.random.seed(42)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_X_y_path = os.path.join(saving_path, "df_X_y.pkl")
config_path = os.path.join(saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)

saved_result = os.path.join(saved_result_path_classification, "ML")
# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)

# Load the dataset and perform preprocessing as previously defined
df = pd.read_pickle(df_X_y_path)
file_name = f"classification_reports_MLP.txt"

# # Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name), "w")

print(f"Shape of the data: {df.shape}")

# Define features and target variable
y = df["gendna_type"]
X = df.drop(columns=["gendna_type", "gendna", "subjid"])

# Load configuration for test subjects
config_dict = pd.read_pickle(config_path)
test_subjects = config_dict["test_subjects"]

# Split data into training and test sets
is_test_subject = df["subjid"].isin(test_subjects)
X_train, y_train = X[~is_test_subject], y[~is_test_subject]
X_test, y_test = X[is_test_subject], y[is_test_subject]

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Apply oversampling to address class imbalance
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

print("Resampled class distribution:", Counter(y_train))

# Evaluate and print classification reports and confusion matrices
def evaluate_and_print_results(model, X_train, y_train, X_test, y_test, activation_func, hidden_layer, learning_rate):
    y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nParameters: \n\tActivation Function: ", activation_func, "\n\tHidden Layer: ", hidden_layer, "\n\tLearning Rate: ", learning_rate)
    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))

    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))


# Define the MLP model architecture
input_dim = X_train.shape[1]
# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Define the hyperparameters to tune for MLP
params = {
    'activation function': ['tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [56/4, 56/2, 56, 56*2, 56*4, 56*8],
    'learning rate': [0.0001, 0.001, 0.01, 0.1]
}

# Evaluate the performance of the MLP model with different hyperparameters
for activation_func in params['activation function']:
    for hidden_layer in params['hidden layer']:
        for learning_rate in params['learning rate']:
            model = Sequential([
                Dense(units=56, input_shape=(input_dim, )),
                Dense(int(hidden_layer), activation=activation_func),
                Dense(1, activation="sigmoid"),
            ])

            # Compile the MLP model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X_train,
                y_train,
                epochs=200,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )

            # Check performance of the improved MLP model
            evaluate_and_print_results(model, X_train, y_train, X_test, y_test, activation_func, hidden_layer, learning_rate)

# Set a random seed for reproducibility
np.random.seed(42)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_X_y_path = os.path.join(saving_path, "df_X_y.pkl")
config_path = os.path.join(saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)

saved_result = os.path.join(saved_result_path_classification, "ML")
# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)

# Load the dataset and perform preprocessing as previously defined
df = pd.read_pickle(df_X_y_path)
file_name = f"classification_reports_MLP.txt"

# # Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name), "w")

print(f"Shape of the data: {df.shape}")

# Define features and target variable
y = df["gendna_type"]
X = df.drop(columns=["gendna_type", "gendna", "subjid"])

# Load configuration for test subjects
config_dict = pd.read_pickle(config_path)
test_subjects = config_dict["test_subjects"]

# Split data into training and test sets
is_test_subject = df["subjid"].isin(test_subjects)
X_train, y_train = X[~is_test_subject], y[~is_test_subject]
X_test, y_test = X[is_test_subject], y[is_test_subject]

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Apply oversampling to address class imbalance
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

print("Resampled class distribution:", Counter(y_train))

input_dim = X_train.shape[1]
# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Define the hyperparameters to tune for MLP
params = {
    'activation function': ['tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [int(56/4), int(56/2), 56, 56*2, 56*4, 56*8],
    'learning rate': [0.0001, 0.001, 0.01, 0.1]
}

# Evaluate and print classification reports and confusion matrices
def evaluate_and_print_results(model, X_train, y_train, X_test, y_test, activation_func, hidden_layer, learning_rate):
    y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nParameters: \n\tActivation Function: ", activation_func, "\n\tHidden Layer: ", hidden_layer, "\n\tLearning Rate: ", learning_rate)
    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))

    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))

# Evaluate the performance of the MLP model with different hyperparameters
for activation_func in params['activation function']:
    for hidden_layer in params['hidden layer']:
        model = Sequential([
            Dense(units=56, input_shape=(input_dim, )),
            Dense(hidden_layer, activation=activation_func),
            Dense(1, activation="sigmoid"),
        ])
        for learning_rate in params['learning rate']:
            # Compile the MLP model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X_train,
                y_train,
                epochs=200,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )

            # Check performance of the improved MLP model
            evaluate_and_print_results(model, X_train, y_train, X_test, y_test, activation_func, hidden_layer, learning_rate)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__

############################################ DNN ########################################################
# Create new file per DNN
file_name_DNN = f"classification_reports_DNN.txt"
print(f"Shape of the data: {df.shape}")
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
print("Resampled class distribution:", Counter(y_train))

# Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name_DNN), "w")

# Define the hyperparameters to tune for DNN
params = {
    'activation function': ['tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [(int(56/2),int(56/4),int(56/8)),(int(56/2),int(56/4)),(56*2,56,int(56/2),int(56/4)),(56*2,56,int(56/2))],
    'learning rate': [0.0001, 0.001, 0.01, 0.1]
}

# Evaluate the performance of the DNN model with different hyperparameters
for activation_func in params['activation function']:
    for hidden_layer in params['hidden layer']:
        model = Sequential()
        model.add(Dense(units=56, input_shape=(input_dim, )))
        for layer in hidden_layer:
            model.add(Dense(layer, activation=activation_func))
        model.add(Dense(1, activation="sigmoid"))
        for learning_rate in params['learning rate']:
            # Compile the MLP model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X_train,
                y_train,
                epochs=200,
                batch_size=64,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )

            # Check performance of the improved MLP model
            evaluate_and_print_results(model, X_train, y_train, X_test, y_test, activation_func, hidden_layer, learning_rate)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__