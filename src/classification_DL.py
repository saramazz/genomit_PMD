# MLP and NN
"""
Code to do the classification using MLP
"""
# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense
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

import tensorflow as tf
from tensorflow import keras
from keras_tuner.tuners import Hyperband

# Set a random seed for reproducibility
np.random.seed(42)

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_X_y_path = os.path.join(saving_path, "df_X_y.pkl")
config_path = os.path.join(
    saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)

saved_result = os.path.join(saved_result_path_classification, "ML")
# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)


# Define the file name with the current date and time
# file_name = f"classification_reports_{current_datetime}_SVM.txt"
file_name = f"classification_reports_ML.txt"

# Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name), "w")


# Load the dataset and perform preprocessing as previously defined
df = pd.read_pickle(df_X_y_path)
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


# Define the MLP model architecture

# Apply oversampling to address class imbalance
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

print("Resampled class distribution:", Counter(y_train))


# Define the MLP model architecture
def build_mlp_simple(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification output layer
    return model


# Improved MLP model architecture
def build_improved_mlp(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))  # Binary classification output layer
    return model

# Compile the improved model
improved_mlp = build_mlp_simple(X_train.shape[1])
improved_mlp.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)


# Evaluate and print classification reports and confusion matrices
def evaluate_and_print_results(model, X_train, y_train, X_test, y_test):
    y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))

    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))

    # Train the improved MLP model with class weights
improved_mlp.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
)



# Check performance of the improved MLP model
evaluate_and_print_results(improved_mlp, X_train, y_train, X_test, y_test)

#Print "starting the analysis with grid search"
print("Starting the analysis with GridSearchCV")

"""
Use the following lines to build, train, and test a MLP model with hyperparameter tuning. 
Remove:

improved_mlp.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
)

in the following to avoid overwritting the model trained with hyperparameter tuning.
"""

################# Modify here FP #################
# Hyperparameter tuning for MLP model (e.g., with GridSearchCV)
# Create a MLP model with not fixed number of hidden layers

#from keras import Hyperband

input_shape = X_train.shape[1]


'''
def build_MLP_model_hp(hp):  # Function to be modified in input parameters (e.g., hp defined outside the function)
    model = Sequential()
    model.add(Dense(input(shape = input_shape)))
    for i in range(hp.Int("hidden_layers", 1, 5)):
        model.add(Dense(units=hp.Int(f"units_{i}",min_value=32,max_value=256, step=32),
                        activation=hp.Choice(f"activation_{i}",values=["relu", "tanh", "sigmoid"])))

    model.add(Dense(1, activation="softmax"))
    model.compile(optimized=Adam(hp.Choice("learning_rate", values=["0.01", "0.001", "0.0001"])),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Instantiate Hyperband Tuner
param_grid = Hyperband(
    build_MLP_model_hp,
    objective="val_accuracy",
    max_epochs=200,
    factor=3,
    directory="mlp_tuning",
    project_name="mlp_tuning"
)
'''
#descrive the function to build the model

def build_MLP_model_hp(hp):
    model = Sequential()

    # Define the input layer
    model.add(Dense(units=hp.Int('input_units', min_value=32, max_value=256, step=32), input_shape=(input_shape,), activation='relu'))

    # Add hidden layers
    for i in range(hp.Int("hidden_layers", 1, 5)):
        model.add(Dense(units=hp.Int(f"units_{i}",min_value=32,max_value=256, step=32),
                        activation=hp.Choice(f"activation_{i}",values=["relu", "tanh", "sigmoid"])))

    # Define the output layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer=Adam(hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

param_grid = Hyperband(build_MLP_model_hp,
                       objective="val_accuracy",
                       max_epochs=200,
                       factor=3,
                       directory="mlp_tuning",
                       project_name="mlp_tuning"
                       )

# Hyperparameter search
param_grid.search(
    X_train, 
    y_train,
    validation_split=0.2,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr])

# Get best hyperparameters
best_hps = param_grid.get_best_hyperparameters(num_trials=1)[0]

# Train the best model
best_model = param_grid.hypermodel.build(best_hps)
improved_mlp = best_model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
)

# Check performance of the improved MLP model
evaluate_and_print_results(improved_mlp, X_train, y_train, X_test, y_test)


################# END FP #################


# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__
