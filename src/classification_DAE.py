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
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers import Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


import sys
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


# import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Set a random seed for reproducibility
np.random.seed(42)

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_X_y_path = os.path.join(saving_path, "df_X_y.pkl")
config_path = os.path.join(
    saved_result_path,
    "classifiers_results/experiments_all_models/classifier_configuration.pkl",
)

saved_result_DAE = os.path.join(saved_result_path_classification, "DAE")
#create the directory for the saved results
if not os.path.exists(saved_result_DAE):
    os.makedirs(saved_result_DAE)

# Redirect the standard output to a file
# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the file name with the current date and time
# file_name = f"classification_reports_{current_datetime}_SVM.txt"
file_name = f"classification_reports_DAE.txt"

# Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result_DAE, file_name), "w")




# Load the dataset
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

'''
OVERSAMPLING
'''
"""
OVERSAMPLING AND CLASSIFICATION
"""
print("Original class distribution:", Counter(y_train))

# Apply oversampling to training data
oversampler = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
X_train, y_train = oversampler.fit_resample(X_train, y_train)


print("Resampled class distribution:", Counter(y_train))


# Define input dimensions
input_dim = X_train.shape[1]


# Define autoencoder model
def build_autoencoder_simple(input_dim):
    input_layer = Input(shape=(input_dim,))
    noisy_input = GaussianNoise(0.1)(input_layer)

    # Encoder
    encoded = Dense(128, activation="relu")(noisy_input)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)

    # Decoder
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    return Model(inputs=input_layer, outputs=decoded)

# Modifying the autoencoder architecture
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    noisy_input = GaussianNoise(0.1)(input_layer)

    # Encoder
    encoded = Dense(256)(noisy_input)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dropout(0.3)(encoded)  # Slightly increasing Dropout

    encoded = Dense(128)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dropout(0.3)(encoded)  # Slightly increasing Dropout

    encoded = Dense(64)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dropout(0.3)(encoded)  # Slightly increasing Dropout

    # Decoder
    decoded = Dense(128)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)

    decoded = Dense(256)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)

    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    return Model(inputs=input_layer, outputs=decoded)

# Training with enhanced callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)



class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # For each epoch end, print the current epoch metrics
        print(f"Epoch {epoch+1}: loss = {logs.get('loss')}, val_loss = {logs.get('val_loss')}")

    def on_train_end(self, logs=None):
        print("Training stopped. Reason for stopping:")
        if 'early_stopping_patience' in self.params.keys():
            print(f"- Early stopping was triggered after patience was reached.")




#if not existing, build and compile the autoencoder
#if not os.path.exists(os.path.join(saved_result_path_classification_models, "denoising_autoencoder.h5")):
print("Building and compiling the autoencoder...")
autoencoder = build_autoencoder(input_dim)
#autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")



# Train the autoencoder
autoencoder.fit(
    X_train,
    X_train,
    epochs=200,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stopping, reduce_lr, CustomCallback()]
)

# Save the trained model in the saved_result_path_classification_models directory
autoencoder.save(
    os.path.join(saved_result_path_classification_models, "denoising_autoencoder.h5")
)
print("Autoencoder saved.")
'''
else:
    print("Loading the autoencoder model...")
    autoencoder = load_model(
    os.path.join(saved_result_path_classification_models, "denoising_autoencoder.h5")
)
'''
    # Extract features from the encoder part of the autoencoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-3].output)

X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)
#print the dimensions of the encoded data
print(f"Encoded training set shape: X_train_encoded={X_train_encoded.shape}")
print(f"Encoded test set shape: X_test_encoded={X_test_encoded.shape}")

# --- Additional Steps for Performance and Feature Importances ---
"""
# Apply oversampling to training data
oversampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_resampled))


print("Starting the classification...")

#best classifier
classifiers = {
    "RandomForestClassifier": (
        RandomForestClassifier(),
        {
            "n_estimators": [100],  # Increased to 3 values
            "max_depth": [20],  # 4 values
            "min_samples_split": [5],  # 3 values
            "min_samples_leaf": [1],  # 3 values
            "max_features": ["sqrt"],  # 2 values
            "bootstrap": [False],  # 2 values
            "criterion": ["gini"],  # 2 values
        },
    )
}

"""

'''
Classifier training and evaluation without encoded features
'''

# Load the important variables from config
kf = config_dict["kf"]
scorer = config_dict["scorer"]
thr = config_dict["thr"]
nFeatures = config_dict["nFeatures"]
num_folds = config_dict["num_folds"]

classifiers = {
    "RandomForestClassifier": (
        RandomForestClassifier(),
        {
            "n_estimators": [100],  # Increased to 3 values
            "max_depth": [20],  # 4 values
            "min_samples_split": [5],  # 3 values
            "min_samples_leaf": [1],  # 3 values
            "max_features": ["sqrt"],  # 2 values
            "bootstrap": [False],  # 2 values
            "criterion": ["gini"],  # 2 values
        },
    )
}


# Selecting the RandomForestClassifier
clf_name, (clf, param_grid) = list(classifiers.items())[0]

# Perform grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=kf,
    scoring=scorer,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
)
grid_search.fit(X_train, y_train)
# Train a Random Forest on encoded features
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf.fit(X_train, y_train)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
clf  = grid_search.best_estimator_ #clf as best estimator
best_score_ = grid_search.best_score_

cv_results = grid_search.cv_results_
print("CV RESULTS:______________________________________________________")
print("Best params:", best_params)
print("Best estimator:", clf)
print("Best score:", best_score_)


# Predict and evaluate
y_pred = clf.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)


# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
confusion_matrix_file = f"cm_DAE"
plot_confusion_matrix(
    y_test,
    y_pred,
    os.path.join(saved_result_DAE, confusion_matrix_file),
)
plt.close()


print("Calculating and Plotting Importances...")
# Feature Importances
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.title("Feature importances based on encoder output")
    plt.bar(
        range(X_train.shape[1]), importances[indices], color="r", align="center"
    )
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(
        os.path.join(saved_result_DAE, "feature_importances_DAE.png")
    )
    plt.close()

# Save the Random Forest model
best_estimator_file = os.path.join(
    saved_result_path_classification_models, "best_DAE_rf_model.pkl"
)
with open(best_estimator_file, "wb") as f:
    pickle.dump(clf, f)
print(f"Best estimator saved to {best_estimator_file}")

# Save results
results_to_save = {
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_mat.tolist(),
    "feature_importances": importances.tolist(),
}
results_file = os.path.join(saved_result_DAE, "results_DAE.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results_to_save, f)

'''
Classifier training and evaluation with encoded features
'''

# Selecting the RandomForestClassifier
clf_name, (clf, param_grid) = list(classifiers.items())[0]

# Perform grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=kf,
    scoring=scorer,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
)
grid_search.fit(X_train_encoded, y_train)
# Train a Random Forest on encoded features
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf.fit(X_train, y_train)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
clf  = grid_search.best_estimator_ #clf as best estimator
best_score_ = grid_search.best_score_

cv_results = grid_search.cv_results_
print("CV RESULTS:______________________________________________________")
print("Best params:", best_params)
print("Best estimator:", clf)
print("Best score:", best_score_)



# Train a Random Forest on encoded features
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf.fit(X_train_encoded, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_encoded)
print("Classification Report:")
print(classification_report(y_test, y_pred))

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)


# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
confusion_matrix_file = f"cm_DAE_encoded"
plot_confusion_matrix(
    y_test,
    y_pred,
    os.path.join(saved_result_DAE, confusion_matrix_file),
)
plt.close()


print("Calculating and Plotting Importances...")
# Feature Importances
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.title("Feature importances based on encoder output")
    plt.bar(
        range(X_train_encoded.shape[1]), importances[indices], color="r", align="center"
    )
    plt.xticks(range(X_train_encoded.shape[1]), indices)
    plt.xlim([-1, X_train_encoded.shape[1]])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(
        os.path.join(saved_result_DAE, "feature_importances_DAE_encoded.png")
    )
    plt.close()

# Save the Random Forest model
best_estimator_file = os.path.join(
    saved_result_path_classification_models, "best_DAE_rf_model_encoded.pkl"
)
with open(best_estimator_file, "wb") as f:
    pickle.dump(clf, f)
print(f"Best estimator saved to {best_estimator_file}")

# Save results
results_to_save = {
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_mat.tolist(),
    "feature_importances": importances.tolist(),
}
results_file = os.path.join(saved_result_DAE, "results_DAE_encoded.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results_to_save, f)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__