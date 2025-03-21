import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif

import pandas as pd
import numpy as np
import pickle
import random
import os

    
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
general_path = os.path.dirname(os.getcwd())

random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

key = input("Enter the classifier:\n1. LR reduced\n2. LR full\n3. NN reduced\n4. NN full\n")
if key == '1':
    result_path = os.path.join(general_path, "Results_14_03\\lr\\reduced\\penalty_factor")
elif key == '2':
    result_path = os.path.join(general_path, "Results_14_03\\lr\\full\\penalty_factor")
elif key == '3':
    result_path = os.path.join(general_path, "Results_14_03\\nn\\reduced\\penalty_factor")
elif key == '4':
    result_path = os.path.join(general_path, "Results_14_03\\nn\\full\\penalty_factor")

if key == '1' or key == '2':
    # Define the model
    class Model(nn.Module):
        def __init__(self, input_dim):
            super(Model, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))
else:

    # Define the model
    class Model(nn.Module):
        def __init__(self, input_dim, hidden_layers, activation_func, dropout):
            super(Model, self).__init__()
            layers = []
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.BatchNorm1d(input_dim))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            for i in range(len(hidden_layers)):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_layers[i]))
                    layers.append(getattr(nn, activation_func)())
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
                else:
                    layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                    layers.append(getattr(nn, activation_func)())
                    layers.append(nn.BatchNorm1d(hidden_layers[i]))
                    if dropout is not None:
                        layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_layers[-1], 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)


all_scores = pd.read_pickle(os.path.join(result_path,"all_scores.pkl"))
all_models = pd.read_pickle(os.path.join(result_path,"all_models.pkl"))
all_conf = pd.read_pickle(os.path.join(result_path,"all_configs.pkl"))

sorted_scores_with_indices = sorted(enumerate(all_scores), key=lambda x: x[1][0], reverse=True)
all_scores = [all_scores[idx] for idx, _ in sorted_scores_with_indices]
all_models = [all_models[idx] for idx, _ in sorted_scores_with_indices]
all_configs = [all_conf[idx] for idx, _ in sorted_scores_with_indices]

# Set path
data_path = os.path.join(general_path, "Data/data_genomit")

# Load the dataset
if key == '1' or key == '3':
    df = pd.read_csv(os.path.join(data_path, "df_reduced.csv"))
else:
    df = pd.read_csv(os.path.join(data_path, "df_full.csv"))

# Swap the values of 'gendna_type' (0 -> 1 and 1 -> 0) to consider mtDNA as the positive class
df['gendna_type'] = df['gendna_type'].apply(lambda x: 1 if x == 0 else 0)

# Remove unnecessary columns
df.drop(['Unnamed: 0', 'subjid'], axis=1, inplace=True)

# Split the data
df_train = df[df['test'] == 0].reset_index(drop=True)
df_test = df[df['test'] == 1].reset_index(drop=True)

# Separate features and target
X_train_full = df_train.drop(['gendna_type', 'test'], axis=1)
y_train_full = df_train['gendna_type']
X_test = df_test.drop(['gendna_type', 'test'], axis=1)
y_test = df_test['gendna_type']

# Compute missing value proportions
missing_ratios = (X_train_full == -998).mean()  # Proportion of missing values per feature
penalty_factors = 1 - missing_ratios  # Create a penalty factor

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_train_scaled_df = (pd.DataFrame(X_train_scaled, columns=X_train_full.columns)) * penalty_factors
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = (pd.DataFrame(X_test_scaled, columns=X_test.columns)) * penalty_factors

# Apply missing value penalty in MRMR feature selection
num_features = X_train_full.shape[1]

X_train_subset = X_train_scaled_df
X_test_subset = X_test_scaled_df

for i in range(len(all_models)):

    best_model = all_models[i]
    best_conf = all_configs[i]
    best_score = all_scores[i]

    if best_conf["sampling"] != 'no_resampling':

        print(f"Best configuration: {best_conf}")
        print(f"Best score: {best_score}")

        if best_conf['features'] != 56:
            print(best_conf['feature set'])
            X_train_subset = X_train_scaled_df[best_conf['feature set']]
            X_test_subset = X_test_scaled_df[best_conf['feature set']]

        # Define resampling strategies
        if best_conf['sampling'] != 'no_resampling':
            if best_conf['sampling'] == 'ADASYN':
                sampler = ADASYN(random_state=random_state)
                # Apply resampling
                X_resampled, y_resampled = X_train_subset, y_train_full
                X_resampled, y_resampled = sampler.fit_resample(X_train_subset, y_train_full)
            else:
                sampler = SMOTE(random_state=random_state)
                # Apply resampling
                X_resampled, y_resampled = X_train_subset, y_train_full
                X_resampled, y_resampled = sampler.fit_resample(X_train_subset, y_train_full)
        else:
            X_resampled, y_resampled = X_train_subset, y_train_full


        best_model.eval()
        with torch.no_grad():
            y_train_pred = (best_model(torch.tensor(X_resampled.values, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()
            accuracy_train = accuracy_score(y_resampled, y_train_pred)
            f1_score_train = f1_score(y_resampled, y_train_pred)
            conf_matrix_train = confusion_matrix(y_resampled, y_train_pred)
            print("######### Triaining Set Performance #########")
            print(conf_matrix_train)
            print(f"Training Accuracy: {accuracy_train:.3f}")
            print(f"F1-score: {f1_score_train:.3f}")


        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_subset.values, dtype=torch.float32).to(device)
            y_pred = (best_model(X_test_tensor).squeeze() > 0.5).int().cpu().numpy()

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print("\nTest Set Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1-score: {f1:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix: {conf_matrix}")
            
            # # Assuming that mtDNA is the positive class, compute sensitivity and specificity
            spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            sens = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
            print(f"Sensitivity: {sens:.3f}")
            print(f"Specificity: {spec:.3f}")

            break
