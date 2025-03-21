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

# Define the model
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
general_path = os.path.dirname(os.getcwd())
result_path_reduced_lr = os.path.join(general_path, "Results_14_03\\lr\\reduced\\penalty_factor")
result_path_full_lr = os.path.join(general_path, "Results_14_03\\lr\\full\\penalty_factor")
result_path_reduced_nn = os.path.join(general_path, "Results_14_03\\nn\\reduced\\penalty_factor")
result_path_full_nn = os.path.join(general_path, "Results_14_03\\nn\\full\\penalty_factor")

def aggregate_results(result_path):

    all_scores_final = []
    all_configs_final = []
    all_models_final = []

    for folder in os.listdir(result_path):
        if not folder.endswith(".pkl"):

            data_path = os.path.join(result_path, folder)
            all_scores = pd.read_pickle(os.path.join(data_path, "all_scores.pkl"))
            all_configs = pd.read_pickle(os.path.join(data_path, "all_configs.pkl"))
            all_models = pd.read_pickle(os.path.join(data_path, "all_models.pkl"))

            for i, (f1_training, accuracy_training) in enumerate(all_scores):
                all_scores_final.append((f1_training, accuracy_training))
                all_configs_final.append(all_configs[i])
                all_models_final.append(all_models[i])

            with open(os.path.join(result_path, "all_scores.pkl"), 'wb') as file:
                pickle.dump(all_scores_final, file)
            with open(os.path.join(result_path, "all_configs.pkl"), 'wb') as file:
                pickle.dump(all_configs_final, file)
            with open(os.path.join(result_path, "all_models.pkl"), 'wb') as file:
                pickle.dump(all_models_final, file)
        
    print("Done!")

aggregate_results(result_path_reduced_lr)
aggregate_results(result_path_full_lr)
aggregate_results(result_path_reduced_nn)
aggregate_results(result_path_full_nn)