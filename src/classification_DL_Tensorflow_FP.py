from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.layers import Dense, Dropout, BatchNormalization
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.optimizers import Adam
from collections import Counter
from datetime import datetime
from itertools import combinations
from sklearn.model_selection import KFold

from config import (global_path, saved_result_path_classification, saved_result_path_classification_models,)
from preprocessing import *
from processing import *
from utilities import *
from plotting import *

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import mrmr
import sys
import os

# Set a random seed for reproducibility
random_state = 42
np.random.seed(random_state)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(random_state)
random.seed(random_state)

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_path = os.path.join(saving_path, "new_dataset.csv")
saved_result = os.path.join(saved_result_path_classification, "ML")

# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)

# Load the dataset and perform preprocessing as previously defined
df = pd.read_csv(df_path)

X = df.drop(columns=["subjid","gendna","test"]).values
y = df["gendna"].values

test = df["test"].values

X_test_tot, y_test = X[test == 1], y[test == 1]
X_train_tot, y_train = X[test == 0], np.array(y)[test == 0]


scaler = MinMaxScaler(feature_range=(-1,1)) 

def normalize(scaler, X_t, X_te):
    X_train_new = scaler.fit_transform(X_t)
    X_test_new = scaler.transform(X_te)
    return X_train_new, X_test_new


def oversampling(state, X, y):
    oversampler = SMOTE(random_state=state)
    X_new, y_new = oversampler.fit_resample(X, y)
    print("Resampled (Oversampled) class distribution:", Counter(y_new))
    return X_new, y_new


def undersampling(state, X, y):
    undersampler = RandomUnderSampler(random_state=state)
    X_new, y_new = undersampler.fit_resample(X, y)
    print("Resampled (Undersampled) class distribution:", Counter(y_new))
    return X_new, y_new


def rankfeatures(X, Y, nFeatures, thr, kf):
    Y = np.asarray(Y)
    top_features_all = []

    # Generate shuffled patient indices
    patient_indices = np.arange(X.shape[0])
    np.random.shuffle(patient_indices)

    # Process K-fold splitting
    for fold, (train_indices, val_indices) in enumerate(kf.split(patient_indices)):
        #print(f"Processing Fold {fold + 1}")
        X_train_cv = X[train_indices]
        y_train_cv = Y[train_indices]

        top_feature_indices = mrmr.mrmr_regression(
            X=pd.DataFrame(X),
            y=pd.DataFrame(Y),
            K=int(thr),
        )
        top_features_all.append(top_feature_indices)

    # Process feature aggregation
    top_features_all_ = np.asarray(top_features_all, dtype=np.float32)
    #print('Found top_features_all_', top_features_all_)

    all_top_features = np.unique(top_features_all_)
    feature_scores = np.empty([len(all_top_features), 2], dtype=int)

    for ii, ff in enumerate(all_top_features):
        idx_ff = np.asarray(np.where(np.isin(top_features_all_, ff)))
        score = np.sum(nFeatures - idx_ff[1])
        feature_scores[ii, :] = [ff, score]

    sorted_scores = np.argsort(feature_scores[:, 1])[::-1]
    top_features = feature_scores[sorted_scores[:int(thr)], 0]
    
    return top_features


num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
features = df.drop(columns=["subjid", "gendna", "test"]).columns
best_report = 0.65 # best f1-score
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0)
model_params = {}
balancing = ["oversampling","undersampling"]
params = {
    'activation function': ['relu','leaky_relu', 'tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [(int(56/8),),
                    (int(56/4),),
                    (int(56/2),),
                    (56,),
                    (56*2,),
                    (int(56/2),int(56/4)),
                    (int(56/2),int(56/4),int(56/8)),
                    (int(56/4),int(56/8),int(56/16)),
                    ],
    'learning rate': [0.0001, 0.001, 0.01, 0.1],
    'dropout': [0.2, 0.3, 0.4, 0.5]
}

for thr in range(10, int(X_train_tot.shape[1] * 0.7), 1):
    # Perform feature selection
    top_features = rankfeatures(X_train_tot, y_train, num_folds, thr, kf)
    top_feature_names = [features[int(i)] for i in top_features]

    X_train_selected = X_train_tot[:, top_features]
    X_test_selected = X_test_tot[:, top_features]

    # Define parameters for DL
    input_dim = X_train_selected.shape[1]
    X_train_selected, X_test_selected = normalize(scaler, X_train_selected, X_test_selected)

    for balance_tech in balancing: 
        if balance_tech == 'oversampling':
            X_train_bal, y_train_bal = oversampling(random_state, X_train_selected, y_train)
        elif balance_tech == 'undersampling':
            X_train_bal, y_train_bal = undersampling(random_state, X_train_selected, y_train)
        
        X_train_bal, X_test_selected = normalize(scaler, X_train_bal, X_test_selected)
        for activation_func in params['activation function']:
            for hidden_layer in params['hidden layer']:
                for learning_rate in params['learning rate']:
                    for drop in params['dropout']:
                        model = Sequential()
                        model.add(Dense(units=input_dim, input_shape=(input_dim, )))
                        model.add(BatchNormalization())
                        model.add(Dropout(drop))
                        for layer in hidden_layer:
                            model.add(Dense(layer, activation=activation_func))
                            model.add(BatchNormalization())
                            model.add(Dropout(drop))
                        model.add(Dense(1, activation="sigmoid"))

                        # Compile the MLP model
                        model.compile(
                            optimizer=Adam(learning_rate=learning_rate),
                            loss="binary_crossentropy",
                            metrics=["accuracy"],
                        )
                        
                        for size in [16, 32]:
                            model.fit(
                                X_train_bal,
                                y_train_bal,
                                epochs=300,
                                batch_size=size,
                                validation_split=0.2,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=0,
                            )
                        
                            y_test_pred = (model.predict(X_test_selected) > 0.5).astype("int32")
                            report = classification_report(y_test, y_test_pred, output_dict=True)
                            if report['weighted avg']['f1-score'] > best_report:
                                best_report = report['weighted avg']['f1-score']
                                model.save(os.path.join(saved_result_path_classification_models, f"best_model.keras"))
                                model_params = {
                                    'activation_func': activation_func,
                                    'hidden_layer': hidden_layer,
                                    'learning_rate': learning_rate,
                                    'dropout': drop,
                                    'balance_tech': balance_tech,
                                    'batch_size': size,
                                    'selected_features': top_features,
                                    'top Feature Names': top_feature_names
                                }
                                y_train_pred = (model.predict(X_train_bal) > 0.5).astype("int32")
                                print(classification_report(y_train_bal, y_train_pred))

model = tf.keras.models.load_model(os.path.join(saved_result_path_classification_models, "best_model.keras"))
X_test = X_test_tot[:, model_params['selected_features']]
X_train = X_train_tot[:, model_params['selected_features']]
X_train, X_test = normalize(scaler, X_train, X_test)
if model_params['balance_tech'] == 'oversampling':
    X_train, y_train = oversampling(random_state, X_train, y_train)
elif model_params['balance_tech'] == 'undersampling':
    X_train, y_train = undersampling(random_state, X_train, y_train)
X_train, X_test = normalize(scaler, X_train, X_test)
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
y_train_pred = (model.predict(X_train) > 0.5).astype("int32")

# Evaluate the performance of the DNN model with different hyperparameters
file_name = f"classification_report.txt"
# # Redirect the standard output to the file
sys.stdout = open(os.path.join(saved_result, file_name), "w")

# print(f"Best f1-score: {best_report}")
print("\nParemeters of the best model:")
print(model_params)

print(f"Shape of the data: {df.shape}")
print("\nTraining Set Performance:")
print(classification_report(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))
print("\nTesting Set Performance:")
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

sys.stdout.close()
sys.stdout = sys.__stdout__