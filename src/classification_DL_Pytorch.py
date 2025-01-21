import torch
import torch.nn as nn
import numpy as np

import sys
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

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

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

######################################## Pytorch Models ########################################
# MLP model
class MLP_model(nn.Module):
    def __init__(self, input_dim, hidden_layer, activation_func):
        super(MLP_model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 56)
        self.activation_func = self.get_activation_func(activation_func)
        self.layer2 = nn.Linear(56, hidden_layer)
        self.layer3 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def get_activation_func(self, activation_func):
        if activation_func == "relu":
            return nn.ReLU()
        elif activation_func == "tanh":
            return nn.Tanh()
        elif activation_func == "sigmoid":
            return nn.Sigmoid()
        elif activation_func == "silu":
            return nn.SiLU()
        elif activation_func == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_func(x)
        x = self.layer2(x)
        x = self.activation_func(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

# DNN model
class DNN_model(nn.Module):
    def __init__(self, input_dim, hidden_layer, activation_func):
        super(DNN_model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 56)
        self.activation_func = self.get_activation_func(activation_func)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(56, hidden_layer[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
        self.layerOut = nn.Linear(hidden_layer[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def get_activation_func(self, activation_func):
        if activation_func == "relu":
            return nn.ReLU()
        elif activation_func == "tanh":
            return nn.Tanh()
        elif activation_func == "sigmoid":
            return nn.Sigmoid()
        elif activation_func == "silu":
            return nn.SiLU()
        elif activation_func == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_func}")
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_func(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_func(x)
        x = self.layerOut(x)
        x = self.sigmoid(x)
        return x
        return x
################################################################################

# Evaluate and print classification reports and confusion matrices
def evaluate_and_print_results(y_train_pred, y_train, y_test_pred, y_test, activation_func, hidden_layer, learning_rate):
    y_train_pred = y_train_pred.cpu().numpy().astype("int32")
    y_train = y_train.cpu().numpy().astype("int32")
    y_test_pred = y_test_pred.cpu().numpy().astype("int32")
    y_test = y_test.cpu().numpy().astype("int32")

    print("\nParameters: \n\tActivation Function: ", activation_func, "\n\tHidden Layer: ", hidden_layer, "\n\tLearning Rate: ", learning_rate)
    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))

    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))

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
file_name = f"classification_reports_MLP_pytorch.txt"

# # Redirect the standard output to the file
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

# Apply oversampling to address class imbalance
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

input_dim = X_train.shape[1]

train_inputs = torch.tensor(X_train.values, dtype=torch.float32).to(device)
train_labels = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
test_inputs = torch.tensor(X_test.values, dtype=torch.float32).to(device)
test_labels = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

######################################## MLP ########################################
# Define the hyperparameters to tune for MLP
params = {
    'activation function': ['tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [
        int(56/4),
        int(56/2),
        56,
        56*2,
        56*4,
        56*8
        ],
    'learning rate': [0.0001, 0.001, 0.01, 0.1]
}

for activation_func in params['activation function']:
    for hidden_layer in params['hidden layer']:
        model = MLP_model(input_dim=input_dim, hidden_layer=hidden_layer, activation_func=activation_func).to(device)
        criterion = nn.BCELoss()
        for learning_rate in params['learning rate']:
            print(f"Evaluating: Activation: {activation_func}, Hidden Layers: {hidden_layer}, Learning Rate: {learning_rate}")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(200):
                # Forward pass
                train_outputs = model(train_inputs)
                loss = criterion(train_outputs, train_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                evaluate_and_print_results(train_outputs,train_labels, test_outputs, test_labels, activation_func, hidden_layer, learning_rate)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__

######################################## DNN ########################################

file_name = f"classification_reports_DNN_pytorch.txt"
sys.stdout = open(os.path.join(saved_result, file_name), "w")
print(f"Shape of the data: {df.shape}")

# Define the hyperparameters to tune for DNN
params = {
    'activation function': ['tanh', 'sigmoid', 'silu', 'gelu'],
    'hidden layer': [
        (int(56/2),int(56/4),int(56/8)),
        (int(56/2),int(56/4)),
        (56*2,56,int(56/2),int(56/4)),
        (56*2,56,int(56/2))
        ],
    'learning rate': [0.0001, 0.001, 0.01, 0.1],
}

for activation_func in params['activation function']:
    for hidden_layer in params['hidden layer']:
        model = DNN_model(input_dim=input_dim, hidden_layer=hidden_layer, activation_func=activation_func).to(device)
        criterion = nn.BCELoss()
        for learning_rate in params['learning rate']:
            print(f"Evaluating: Activation: {activation_func}, Hidden Layers: {hidden_layer}, Learning Rate: {learning_rate}")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(200):
                model.train()  # Set the model to training mode
                train_outputs = model(train_inputs)
                loss = criterion(train_outputs, train_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Evaluate the model on the test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                evaluate_and_print_results(train_outputs,train_labels, test_outputs, test_labels, activation_func, hidden_layer, learning_rate)

# Close the file and restore the standard output
sys.stdout.close()
sys.stdout = sys.__stdout__