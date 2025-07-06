import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os

# Define file paths
gpt4o_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_gpt-4o.csv"
sauerkraut_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_sauerkrautlm-gemma-2-9b-it-i1.csv"
gold_path = "/home/saram/PhD/genomit_PMD/saved_results/df/df_symp.csv"
output_dir = "/home/saram/PhD/genomit_PMD/saved_results/survey"
phi4_mini_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_phi-4-mini-reasoning.csv"
deepseek_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_deepseek-r1-distill-qwen-7b.csv"
gpt_35_turbo_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_gpt-3.5-turbo.csv"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load gold standard data
gold_df = pd.read_csv(gold_path)
gold_df.rename(columns={"subjid": "ID"}, inplace=True)
gold_df = gold_df[["ID", "gendna_type"]]

# List of models and their corresponding file paths
models = {
    "gpt-4o": gpt4o_path,
    "gpt-3.5-turbo": gpt_35_turbo_path,  # Assuming gpt-3.5-turbo uses the same file as gpt-4o
    "sauerkrautlm-gemma-2-9b-it-i1": sauerkraut_path,
    "phi-4-mini-reasoning": phi4_mini_path,
    "deepseek-r1-distill-qwen-7b": deepseek_path,
}

# Initialize a list to store performance metrics
metrics_list = []


# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()


# Iterate over each model
for model_name, model_path in models.items():
    # Load model predictions
    model_df = pd.read_csv(model_path)

    #remove rows with NaN values in 'mutation' column
    model_df = model_df.dropna(subset=["mutation"])

    # Merge with gold standard
    merged_df = pd.merge(model_df, gold_df, on="ID", suffixes=("_pred", "_true"))

    # Extract predictions and true labels
    y_pred = merged_df["mutation"]
    y_true = merged_df["gendna_type"]

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    # Append metrics to the list
    metrics_list.append(
        {
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
        }
    )

    print(f"Performance for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, model_name, output_dir)

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Save metrics to CSV
metrics_df.to_csv(os.path.join(output_dir, "performance_comparison.csv"), index=False)

print("Performance evaluation completed. Results saved to the specified directory.")
