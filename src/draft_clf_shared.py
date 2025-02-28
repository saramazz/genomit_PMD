import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils.class_weight import compute_class_weight
import shap
from mrmr import mrmr_classif
import xgboost as xgb
from sklearn.svm import SVC

#%%
# Load and preprocess the data
df = pd.read_csv('/Users/andrea/Desktop/df_symp.csv')

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
selected_features = mrmr_classif(X_train_scaled_df, y_train_full, K=num_features)

# Create feature sets in increasing length order
feature_sets = [selected_features[:i] for i in range(1, len(selected_features) + 1)]

# Define resampling strategies
samplers = {
    "no_resampling": None,
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42)
}

# Define hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [2, 5, 10, 20]
}

# Define hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 5, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define hyperparameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store all models and configurations
all_scores = []
all_models = []
all_configs = []

# Select the model to optimize
model_choice = input("Choose a model to optimize (RF/XGB/SVM): ").strip().lower()

# Total iterations count
total_iterations = len(samplers) * len(feature_sets)
iteration_count = 0

# Loop through feature sets and sampling methods
for feature_set in feature_sets:
    X_train_subset = X_train_scaled_df[feature_set]
    X_test_subset = X_test_scaled_df[feature_set]

    for sampling_name, sampler in samplers.items():
        iteration_count += 1
        print(f"\nIteration {iteration_count}/{total_iterations} | Features: {len(feature_set)} | Sampling: {sampling_name}")

        # Apply resampling
        X_resampled, y_resampled = X_train_subset, y_train_full
        if sampler:
            X_resampled, y_resampled = sampler.fit_resample(X_train_subset, y_train_full)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

        # Choose the model based on user input
        if model_choice == 'rf':
            model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
            grid_search = GridSearchCV(model, param_grid_rf, cv=cv, scoring='f1', n_jobs=-1)
        elif model_choice == 'xgb':
            model = xgb.XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[1])
            grid_search = GridSearchCV(model, param_grid_xgb, cv=cv, scoring='f1', n_jobs=-1)
        elif model_choice == 'svm':
            model = SVC(class_weight='balanced', random_state=42)
            grid_search = GridSearchCV(model, param_grid_svm, cv=cv, scoring='f1', n_jobs=-1)
        else:
            raise ValueError("Invalid model choice. Choose either 'RF', 'XGB', or 'SVM'.")

        # Fit the model
        grid_search.fit(X_resampled, y_resampled)

        # Print metrics during training-validation loop
        y_pred_train = grid_search.best_estimator_.predict(X_resampled)
        accuracy_train = accuracy_score(y_resampled, y_pred_train)
        f1_score_train = f1_score(y_resampled, y_pred_train)
        conf_matrix_train = confusion_matrix(y_resampled, y_pred_train)
        print(conf_matrix_train)
        print(f"Training Accuracy: {accuracy_train:.3f}")
        print(f"F1-score: {f1_score_train:.3f}")
        
        # Save all models and configurations
        all_scores.append((f1_score_train, accuracy_train))
        all_models.append(grid_search.best_estimator_)
        all_configs.append({
            "feature set": feature_set,
            "features": len(feature_set),
            "sampling": sampling_name,
            "model": model_choice.upper(),
            "hyperparameters": grid_search.best_params_
        })

#%%
# Sort the scores in descending order based on the F1-score (first value in tuple), keeping corresponding accuracy (second value)
sorted_scores_with_indices = sorted(enumerate(all_scores), key=lambda x: x[1][0], reverse=True)

# Ask user to choose the best model for testing
print("\nChoose the best model configuration for the test set:")
for idx, (original_idx, (f1_training, accuracy_training)) in enumerate(sorted_scores_with_indices):
    print(f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}")

model_idx = int(input("Enter the index number of the model to evaluate on the test set: ")) - 1

# Evaluate selected model
selected_model = all_models[model_idx]
selected_config = all_configs[model_idx]
print("\nEvaluating Best Model on Test Set:")

y_pred = selected_model.predict(X_test_scaled_df[selected_config['feature set']])

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
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["nDNA", "mtDNA"], yticklabels=["nDNA", "mtDNA"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Selected Model")
plt.show()

# Assuming that mtDNA is the positive class, compute sensitivity and specificity
spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sens = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
print(f"Sensitivity: {sens:.3f}")
print(f"Specificity: {spec:.3f}")