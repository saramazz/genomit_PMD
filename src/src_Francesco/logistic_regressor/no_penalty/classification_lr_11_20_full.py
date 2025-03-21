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
print(device)

# Set a random seed for reproducibility
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set path
general_path = os.path.dirname(os.getcwd())
data_path = os.path.join(general_path, "Data/data_genomit")
result_path = os.path.join(general_path, "Results_14_03/lr/full/no_penalty_factor/11_20")
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Load the dataset
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
X_train_scaled_df = (pd.DataFrame(X_train_scaled, columns=X_train_full.columns)) 
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = (pd.DataFrame(X_test_scaled, columns=X_test.columns)) 

# Apply missing value penalty in MRMR feature selection
num_features = X_train_full.shape[1]
selected_features = mrmr_classif(X_train_scaled_df, y_train_full, K=num_features)

# Create feature sets in increasing length order
feature_sets = [selected_features[:i] for i in range(1, len(selected_features) + 1)]

# Define resampling strategies
samplers = {
    "no_resampling": None,
    "SMOTE": SMOTE(random_state=random_state),
    "ADASYN": ADASYN(random_state=random_state)
}

params_grid = {
    'threshold': [0.5, 0.55, 0.6, 0.65, 0.7],
    'learning rate': [0.0001, 0.001],
    'batch_size': [8, 16]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# Store all models and configurations
all_scores = []
all_models = []
all_configs = []

# Define the model
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# class EarlyStopping:
#     def __init__(self, patience=10, restore_best_weights=True):
#         self.patience = patience
#         self.restore_best_weights = restore_best_weights
#         self.best_loss = float('inf')
#         self.best_model_state = None
#         self.epochs_since_improvement = 0

#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss:
#             self.best_loss = val_loss
#             self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             self.epochs_since_improvement = 0  # Reset patience counter
#         else:
#             self.epochs_since_improvement += 1

#         if self.epochs_since_improvement >= self.patience:
#             if self.restore_best_weights and self.best_model_state:
#                 model.load_state_dict(self.best_model_state)
#             return True  # Stop training

#         return False  # Continue training

# class ReduceLROnPlateau:
#     def __init__(self, optimizer, factor=0.2, patience=5, min_lr=0.0001):
#         self.optimizer = optimizer
#         self.factor = factor
#         self.patience = patience
#         self.min_lr = min_lr
#         self.best_loss = float('inf')
#         self.epochs_since_improvement = 0

#     def __call__(self, val_loss):
#         if val_loss < self.best_loss:
#             self.best_loss = val_loss
#             self.epochs_since_improvement = 0  # Reset patience counter
#         else:
#             self.epochs_since_improvement += 1

#         if self.epochs_since_improvement >= self.patience:
#             for param_group in self.optimizer.param_groups:
#                 new_lr = max(param_group['lr'] * self.factor, self.min_lr)
#                 if param_group['lr'] > new_lr:
#                     param_group['lr'] = new_lr
#             self.epochs_since_improvement = 0  # Reset patience counter

    
# Loop through feature sets and sampling methods
for feature_set in feature_sets[10:20]:

    X_train_subset = X_train_scaled_df[feature_set]
    X_test_subset = X_test_scaled_df[feature_set]
    input_dim = X_train_subset.shape[1]

    for sampling_name, sampler in samplers.items():

        # Apply resampling
        X_resampled, y_resampled = X_train_subset, y_train_full
        if sampler:
            X_resampled, y_resampled = sampler.fit_resample(X_train_subset, y_train_full)

        # Compute class weights
        # class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        # class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

        # Create a model and grid search
        for learning_rate in params_grid['learning rate']:
            for threshold in params_grid['threshold']:
                print(f'Learning rate: {learning_rate} | threshold: {threshold}')
                model = Model(input_dim).to(device)

                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                # reduce_lr = ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=0.0001)

                fold_scores = []
                best_model = None
                
                max_f1_score = 0

                for train_idx, val_idx in cv.split(X_resampled, y_resampled):
                    X_train_split = torch.tensor(X_resampled.iloc[train_idx].values, dtype=torch.float32).to(device)
                    y_train_split = torch.tensor(y_resampled.iloc[train_idx].values, dtype=torch.float32).to(device)
                    X_val_split = torch.tensor(X_resampled.iloc[val_idx].values, dtype=torch.float32).to(device)
                    y_val_split = torch.tensor(y_resampled.iloc[val_idx].values, dtype=torch.float32).to(device)
                    
                    for batch_size in params_grid['batch_size']:

                        train_loader = DataLoader(TensorDataset(X_train_split, y_train_split), batch_size=batch_size,drop_last=True, shuffle=True)
                        val_loader = DataLoader(TensorDataset(X_val_split, y_val_split), batch_size=batch_size, drop_last=True)

                        for epoch in range(100):
                            model.train()
                            for X_batch, y_batch in train_loader:
                                optimizer.zero_grad()
                                outputs = model(X_batch).squeeze()
                                loss = criterion(outputs, y_batch.float())
                                loss.backward()
                                optimizer.step()

                            # Validation
                            model.eval()
                            y_true, y_pred = [], []
                            # validation_loss = 0.0
                            with torch.no_grad():
                                for X_val, y_val in val_loader:
                                    X_val, y_val = X_val.to(device), y_val.to(device)
                                    probs = model(X_val).cpu().numpy()
                                    outputs = (probs > threshold).astype(int)
                                    y_true.extend(y_val.cpu().numpy())
                                    y_pred.extend(outputs) 
                            f1 = f1_score(y_true, y_pred)
                                    
                            #         loss = criterion(outputs, y_val.float())
                            #         preds = (outputs > 0.5).int()
                            #         y_true.extend(y_val.cpu().numpy())
                            #         y_pred.extend(preds.cpu().numpy())
                            #         validation_loss += loss.item()                                    
                            # validation_loss /= len(val_loader)

                            # f1 = f1_score(y_true, y_pred)

                            # if not early_stopping(validation_loss, model):
                            #     reduce_lr(validation_loss)
                            # else:
                            #     break
                            
                            # best for this fold
                            if f1 >= max_f1_score:
                                best_model = model
                                max_f1_score = f1
                    
                del model
                torch.cuda.empty_cache()
                
                # Evaluate the model on the trsin set
                best_model.eval()
                with torch.no_grad():
                    train_pred = best_model(torch.tensor(X_resampled.values, dtype=torch.float32).to(device)).cpu().numpy()
                    y_train_pred = (train_pred > threshold).astype(int)
                    accuracy_train = accuracy_score(y_resampled, y_train_pred)
                    f1_score_train = f1_score(y_resampled, y_train_pred)
                    conf_matrix_train = confusion_matrix(y_resampled, y_train_pred)
                    print(f"Training Accuracy: {accuracy_train:.3f} | F1-score: {f1_score_train:.3f}")

                    # Save all models and configurations
                    all_scores.append((f1_score_train, accuracy_train))
                    all_models.append(best_model)
                    all_configs.append({
                        "feature set": feature_set,
                        "features": len(feature_set),
                        "sampling": sampling_name,
                        "confusion_matrix": conf_matrix_train.tolist(),
                        "hyperparameters": {
                            "learning rate": learning_rate,
                            "batch_size": batch_size,
                            "threshold": threshold
                        }
                    })

                    # Save all models and configurations
                    with open(os.path.join(result_path, "all_scores.pkl"), 'wb') as file:
                        pickle.dump(all_scores, file)
                    with open(os.path.join(result_path, "all_configs.pkl"), 'wb') as file:
                        pickle.dump(all_configs, file)
                    with open(os.path.join(result_path, "all_models.pkl"), 'wb') as file:
                        pickle.dump(all_models, file)

                del best_model
                torch.cuda.empty_cache()