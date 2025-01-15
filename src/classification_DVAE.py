# Ricostruzione pazienti mancanti con denoising VAE addestrato sui 714 pazienti di training e aggiunta al training set (@Sara Mazzucato @Francesco Pierotti)
### Script for Reconstructing Missing Patients with Denoising VAE
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

# Import necessary libraries
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from keras import backend as K
from keras.losses import MeanSquaredError

mse = MeanSquaredError()

from sklearn.preprocessing import StandardScaler
import sys
from datetime import datetime

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
vae_model_path = os.path.join(
    saved_result_path_classification_models, "denoising_vae.h5"
)

saved_result = os.path.join(saved_result_path_classification, "DVAE")
# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)


# Redirect the standard output to a file
# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the file name with the current date and time
# file_name = f"classification_reports_{current_datetime}_SVM.txt"
file_name = f"classification_reports_DvAE.txt"

# Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name), "w")


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

# Apply oversampling to address class imbalance
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

print("Resampled class distribution:", Counter(y_train))


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# VAE Hyperparameters
latent_dim = 2  # Hyperparameter to change based on dataset complexity


# Sampling Layer
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# VAE Model Definition
inputs = Input(shape=(X_train_scaled.shape[1],), name="encoder_input")
x = Dense(128, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)

# Sampling Layer
z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Encoder Model
encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder Model
latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
x = Dense(64, activation="relu")(latent_inputs)
x = Dense(128, activation="relu")(x)
outputs = Dense(X_train_scaled.shape[1], activation="sigmoid")(x)

# Define VAE model
decoder = Model(latent_inputs, outputs, name="decoder")
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name="vae_mlp")

# VAE Loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= X_train_scaled.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer=Adam())

# Train the VAE
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

vae.fit(
    X_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
)

vae.save(vae_model_path)
print("VAE model saved.")

# Reconstruction of Missing Patients
# Assume the test set represents missing patients (just for demonstration)
X_test_encoded, _, _ = encoder.predict(X_test_scaled)
X_test_reconstructed = decoder.predict(X_test_encoded)

# Reverse scaling
X_test_reconstructed = scaler.inverse_transform(X_test_reconstructed)

# Append reconstructed patients to the training set
X_train_augmented = np.concatenate([X_train, X_test_reconstructed], axis=0)
y_train_augmented = np.concatenate([y_train, y_test], axis=0)

print(
    f"Augmented training dataset shape: {X_train_augmented.shape}, {y_train_augmented.shape}"
)

# Train the classifiers with the augmented dataset
# Define the classifiers and their parameter grids
"""
CLASSIFICATION
"""

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
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
clf = grid_search.best_estimator_  # clf as best estimator
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
confusion_matrix_file = f"cm_DVAE"
plot_confusion_matrix(
    y_test,
    y_pred,
    os.path.join(saved_result, confusion_matrix_file),
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
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(os.path.join(saved_result, "feature_importances_DVAE.png"))
    plt.close()

# Save the Random Forest model
best_estimator_file = os.path.join(
    saved_result_path_classification_models, "best_DVAE_rf_model.pkl"
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
results_file = os.path.join(saved_result, "results_DVAE.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results_to_save, f)

"""
Classifier training and evaluation with augmentation
"""

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
grid_search.fit(X_train_augmented, y_train_augmented)  # chek it
# Train a Random Forest on encoded features
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# Get the best parameters and best estimator from grid search
best_params = grid_search.best_params_
clf = grid_search.best_estimator_  # clf as best estimator
best_score_ = grid_search.best_score_

cv_results = grid_search.cv_results_
print("CV RESULTS:______________________________________________________")
print("Best params:", best_params)
print("Best estimator:", clf)
print("Best score:", best_score_)


# Train a Random Forest on encoded features
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_augmented, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)


# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
confusion_matrix_file = f"cm_DVAE_encoded"
plot_confusion_matrix(
    y_test,
    y_pred,
    os.path.join(saved_result, confusion_matrix_file),
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
        range(X_train_augmented.shape[1]),
        importances[indices],
        color="r",
        align="center",
    )
    plt.xticks(range(X_train_augmented.shape[1]), indices)
    plt.xlim([-1, X_train_augmented.shape[1]])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(os.path.join(saved_result, "feature_importances_DVAE_encoded.png"))
    plt.close()

# Save the Random Forest model
best_estimator_file = os.path.join(
    saved_result_path_classification_models, "best_DVAE_rf_model_encoded.pkl"
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
results_file = os.path.join(saved_result, "results_DVAE_encoded.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results_to_save, f)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__
