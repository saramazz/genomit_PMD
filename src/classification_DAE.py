"""
This section of the script implements the training and optimization of a denoising autoencoder, 
specifically tailored for preprocessing a data with  training patients from the current version. 
The autoencoder aims to enhance the data quality by reducing noise, which can improve later classification performance.

1. **Architecture Design**:
   - Constructs an autoencoder neural network using Keras, featuring an input layer with Gaussian noise to introduce robustness.
   - The encoder path compresses the input data using progressively smaller Dense layers.
   - The decoder reconstructs the data to its original dimension, simulating denoising.

2. **Training Process**:
   - Compiles the model with mean squared error as the loss function and the Adam optimizer, common choices for denoising tasks.
   - Trains the model using the input data as both features and labels, leveraging 20% of the data for validation to mitigate overfitting.
   - Configures parameters such as epochs and batch size to balance training speed and performance.

3. **Output and Storage**:
   - Saves the trained autoencoder model as 'denoising_autoencoder.h5' for future application or further refinement.
   - Designed for reuse with similar data structures, enhancing modularity and extensibility in machine learning pipelines.
"""

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

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    LeaveOneGroupOut,
    KFold,
)


# Local imports
from config import global_path, saved_result_path_classification
from utilities import *
from preprocessing import *
from processing import *
from plotting import *

# Set a random seed for reproducibility
np.random.seed(42)

# Keras imports
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam


config_path = os.path.join(
    saved_result_path, "classifiers_results/experiments/classifier_configuration.pkl"
)
config_dict = pd.read_pickle(config_path)

test_subjects = config_dict["test_subjects"]

# Creating the training and test sets
test_set = df[df["subjid"].isin(test_subjects)]
train_set = df[~df["subjid"].isin(test_subjects)]

X_train = train_set.drop(columns=["subjid", "gendna_type", "gendna"])
y_train = train_set["gendna_type"]

X_test = test_set.drop(columns=["subjid", "gendna_type", "gendna"])
y_test = test_set["gendna_type"]

features = X_train.columns

print("X_train shape:", X_train.shape)


# Load the important variables from config
kf = config_dict["kf"]
scorer = config_dict["scorer"]
thr = config_dict["thr"]
nFeatures = config_dict["nFeatures"]
num_folds = config_dict["num_folds"]


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Load the training data
X_train = np.load("X_train.npy")

# Ensure that X_train is loaded correctly and define input dimensions
input_dim = X_train.shape[1]

# Autoencoder definition
input_layer = Input(shape=(input_dim,))
# Add Gaussian noise to input
noisy_input = GaussianNoise(0.1)(input_layer)

# Encoder architecture
encoded = Dense(128, activation="relu")(noisy_input)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

# Decoder architecture
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

# Model definition
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the autoencoder
autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,  # Reserve a subset of data for validation
    verbose=2,
)

# Save the trained autoencoder model
autoencoder.save("denoising_autoencoder.h5")
