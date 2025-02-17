import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from datetime import datetime
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
import torch.multiprocessing as mp

from config import (global_path, saved_result_path_classification, saved_result_path_classification_models,)
from preprocessing import *
from processing import *
from utilities import *
from plotting import *

import pandas as pd
import numpy as np
import random
import json
import ast
import sys
import re
import os

# Set a random seed for reproducibility
random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)
random.seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data paths
saving_path = os.path.join(saved_result_path_classification, "saved_data")
df_path = os.path.join(saving_path, "dataset_11_02.csv")
saved_result = os.path.join(saved_result_path_classification, "ML")

# create the directory for the saved results
if not os.path.exists(saved_result):
    os.makedirs(saved_result)

# Load the dataset and perform preprocessing as previously defined
df = pd.read_csv(df_path)

X = df.drop(columns=["subjid","gendna","test","Unnamed: 0"]).values
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

class Model(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_func, dropout):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Dropout(dropout))
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_layers[i]))
                layers.append(getattr(nn, activation_func)())
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                layers.append(getattr(nn, activation_func)())
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(X_train_bal, y_train_bal, X_val_split, y_val_split, X_test_selected, y_test, input_dim, hidden_layer, activation_func, learning_rate, drop, device, best_report, balance_tech, top_features, top_feature_names, model_params, saved_result_path_classification_models):
    model = Model(input_dim, hidden_layer, activation_func, drop).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(torch.tensor(X_train_bal, dtype=torch.float32), torch.tensor(y_train_bal, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val_split, dtype=torch.float32), torch.tensor(y_val_split, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    for epoch in range(200):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
    
    model.eval()
    with torch.no_grad():
        y_test_pred = (model(torch.tensor(X_test_selected, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()
        report = classification_report(y_test, y_test_pred, output_dict=True)
        if report['weighted avg']['f1-score'] > best_report.value:
            best_report.value = report['weighted avg']['f1-score']
            torch.save(model.state_dict(), os.path.join(saved_result_path_classification_models, f"best_model.pth"))
            model_params.update({
                'activation_function': [activation_func],
                'hidden_layer': [hidden_layer],
                'learning_rate': [learning_rate],
                'dropout': [drop],
                'balance_tech': [balance_tech],
                'selected_features': [top_features],
                'top_feature_names': [top_feature_names]
            })
            y_train_pred = (model(torch.tensor(X_train_bal, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()
            report_train = classification_report(y_train_bal, y_train_pred, output_dict=True)
            print(model_params)
            par = dict(model_params)
            df_params = pd.DataFrame(par)
            df_params.to_csv(os.path.join(saved_result_path_classification_models, "best_model_params.csv"))
            print(f"f1-score on train: {report_train['weighted avg']['f1-score']}")
            print(f'Best f1-score on test: {best_report.value}')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    features = df.drop(columns=["subjid", "gendna", "test"]).columns
    best_report = mp.Value('d', 0.65) # best f1-score
    balancing = ["oversampling","undersampling"]
    params = {
        'activation function': ['ReLU','LeakyReLU', 'Tanh', 'Sigmoid', 'SiLU', 'GELU'],
        'hidden layer': [(int(56/8),),
                        (int(56/4),),
                        (int(56/2),),
                        (56,),
                        (int(56/2),int(56/4)),
                        (int(56/2),int(56/4),int(56/8)),
                        (int(56/4),int(56/8),int(56/16)),
                        ],
        'learning rate': [0.001, 0.01, 0.1],
        'dropout': [0.2, 0.3]
    }
    manager = mp.Manager()
    model_params = manager.dict()
    input_dim = 0
    model_params = {}
    for thr in range(10, int(X_train_tot.shape[1] * 0.5), 5):
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
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_bal, y_train_bal, test_size=0.2, random_state=random_state)

            X_train_bal, X_test_selected = normalize(scaler, X_train_bal, X_test_selected)
            processes = []
            for activation_func in params['activation function']:
                for hidden_layer in params['hidden layer']:
                    for learning_rate in params['learning rate']:
                        for drop in params['dropout']:
                            p = mp.Process(target=train_model, args=(X_train_bal, y_train_bal, X_val_split, y_val_split, X_test_selected, y_test, input_dim, hidden_layer, activation_func, learning_rate, drop, device, best_report, balance_tech, top_features, top_feature_names, model_params, saved_result_path_classification_models))
                            p.start()
                            processes.append(p)
            for p in processes:
                p.join()

    parameters = pd.read_csv(os.path.join(saved_result_path_classification_models, "best_model_params.csv"))
    hidden_layer = ast.literal_eval(parameters['hidden_layer'][0])
    activation_func = parameters['activation_function'][0]
    dropout = float(parameters['dropout'][0])
    balance_tech = parameters['balance_tech'][0]
    selected_features_str = parameters['selected_features'][0]
    selected_features_str = re.sub(r"\s+", ", ", selected_features_str.strip())
    top_features = ast.literal_eval(selected_features_str)
    top_feature_names = ast.literal_eval(parameters['top_feature_names'][0])
    print(len(top_feature_names))
    model = Model(len(top_features), hidden_layer, activation_func, dropout).to(device)
    model.load_state_dict(torch.load(os.path.join(saved_result_path_classification_models, "best_model.pth")))
    X_test = X_test_tot[:, top_features]
    X_train = X_train_tot[:, top_features]
    X_train, X_test = normalize(scaler, X_train, X_test)
    if balance_tech == 'oversampling':
        X_train, y_train = oversampling(random_state, X_train, y_train)
    elif balance_tech == 'undersampling':
        X_train, y_train = undersampling(random_state, X_train, y_train)
    X_train, X_test = normalize(scaler, X_train, X_test)
    y_test_pred = (model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()
    y_train_pred = (model(torch.tensor(X_train, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()

    # Evaluate the performance of the DNN model with different hyperparameters
    file_name = f"classification_report.txt"
    # # Redirect the standard output to the file
    sys.stdout = open(os.path.join(saved_result, file_name), "w")

    # print(f"Best f1-score: {best_report}")
    print("\nParemeters of the best model:")
    model_params = dict({
        'activation_function': activation_func,
        'hidden_layer': hidden_layer,
        'balance_tech': balance_tech,
        'dropout': dropout,
        'selected_features': top_features,
        'top_feature_names': top_feature_names
    })
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