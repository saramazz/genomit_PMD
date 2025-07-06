'''
Script to evaluate the performance of different LLMs on a survey dataset
by comparing their predictions against a gold standard.
Plots confusion matrices and saves performance metrics to a CSV file.
Plots confidence score distributions by true class labels for each model.

'''

# Standard libraries
import os
from pathlib import Path

# Data handling
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn metrics for evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Define your clinical diagnosis mapping at the top level
CLINDIAG_MAPPING = {
    "A01": "LHON",
    "A02": "ADOA",
    "A03": "Other MON",
    "B01": "CPEO",
    "B02": "CPEO Plus",
    "B03": "MiMy (no PEO)",
    "C01": "MELAS",
    "C02": "MIDD",
    "C03": "MERRF",
    "C04": "Leigh",
    "C05": "NARP",
    "C06": "KSS",
    "C07": "SANDO/MIRAS/SCAE",
    "C08": "MNGIE",
    "C09": "Pearson",
    "C12": "Wolfram",
    "C16": "LBSL",
    "C17": "Other MMSD",
    "C18": "Encephalomyopathy",
    "C19": "Encephalopathy",
    "C":   "MMSD",
    "D01": "Cardiomyopathy",
    "D05": "Other MOD",
    "E":   "Unspecified MD",
    "F":   "Asymptomatic"
}



def classify_confusion(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "TP"
    elif true_label == 0 and pred_label == 0:
        return "TN"
    elif true_label == 0 and pred_label == 1:
        return "FP"
    elif true_label == 1 and pred_label == 0:
        return "FN"
    else:
        return "Unknown"

def analyze_misclassifications(
    merged_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    clindiag_mapping: dict,
    clinical_diag_path: str,
):
    # Add confusion labels column
    merged_df["confusion_label"] = merged_df.apply(
        lambda row: classify_confusion(row["gendna_type"], row["mutation"]), axis=1
    )

    # Load clinical diagnosis data and merge
    clinical_df = pd.read_csv(clinical_diag_path)
    merged_df = pd.merge(
        merged_df,
        clinical_df[["subjid", "clindiag__decod"]],
        left_on="ID",
        right_on="subjid",
        how="left"
    )

    # Map clinical diagnosis codes to descriptions
    merged_df["clindiag__decod"] = merged_df["clindiag__decod"].map(clindiag_mapping)

    print(f"Applied clindiag__decod mapping for model {model_name}")

    # Filter False Positives and False Negatives
    fp_df = merged_df[merged_df["confusion_label"] == "FP"]
    fn_df = merged_df[merged_df["confusion_label"] == "FN"]

    # Calculate percentages for each clinical diagnosis category
    fp_percent = (fp_df["clindiag__decod"].value_counts(normalize=True) * 100).sort_index()
    fn_percent = (fn_df["clindiag__decod"].value_counts(normalize=True) * 100).sort_index()



    # Plot FP and FN distributions side-by-side
    plt.figure(figsize=(20, 10))

    # Plot FP if not empty
    if not fp_percent.empty:
        plt.subplot(1, 2, 1)
        fp_percent.sort_values(ascending=False).plot(kind="bar", color="#1f77b4")
        plt.xlabel("Clinical Diagnosis", fontsize=16)
        plt.ylabel("Percentage (%)", fontsize=16)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.title("False Positives (FP)", fontsize=18)
    else:
        print(f"No false positives to plot for model {model_name}")

    # Plot FN if not empty
    if not fn_percent.empty:
        plt.subplot(1, 2, 2)
        fn_percent.sort_values(ascending=False).plot(kind="bar", color="#aec7e8")
        plt.xlabel("Clinical Diagnosis", fontsize=16)
        plt.ylabel("Percentage (%)", fontsize=16)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.title("False Negatives (FN)", fontsize=18)
    else:
        print(f"No false negatives to plot for model {model_name}")

    # Save plot if there is something to show
    if not fp_percent.empty or not fn_percent.empty:
        plt.tight_layout()
        fpfn_dir = os.path.join(output_dir, "FP_FN")
        os.makedirs(fpfn_dir, exist_ok=True)
        plt.savefig(os.path.join(fpfn_dir, f"fp_fn_distribution_{model_name}.png"), dpi=300)
        plt.close()
        print(f"Saved FP/FN plot for model {model_name}")


# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm_dir = os.path.join(output_dir, "CM")
    os.makedirs(cm_dir, exist_ok=True)

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

    plt.savefig(os.path.join(cm_dir, f"confusion_matrix_{model_name}.png"))
    plt.close()

def plot_confidence_distribution_by_class_side_by_side(df, true_label_col, pred_label_col, confidence_score_col, model_name, output_dir):
    """
    Plot KDE distributions of confidence scores side by side:
    - Left: by true class labels
    - Right: by predicted class labels
    
    Parameters:
        df (pd.DataFrame): DataFrame containing true labels, predicted labels, and confidence scores.
        true_label_col (str): Column name for true class labels.
        pred_label_col (str): Column name for predicted class labels.
        confidence_score_col (str): Column name for confidence scores.
        model_name (str): Model name for titles and saving.
        output_dir (str): Directory to save the plot.
    """
    score_dir = os.path.join(output_dir, "scores")
    os.makedirs(score_dir, exist_ok=True)

    colors = {
        0: "#D62728",  # red for class 0 (nDNA)
        1: "#1F77B4",  # blue for class 1 (mtDNA)
    }

    plt.figure(figsize=(14, 6))

    # Plot KDE by TRUE labels (left)
    plt.subplot(1, 2, 1)
    for cls in [0, 1]:
        sns.kdeplot(
            data=df[df[true_label_col] == cls],
            x=confidence_score_col,
            fill=True,
            common_norm=False,
            alpha=0.5,
            label=f"Class {cls} ({'nDNA' if cls == 0 else 'mtDNA'})",
            color=colors[cls],
        )
        mean_val = df[df[true_label_col] == cls][confidence_score_col].mean()
        plt.axvline(mean_val, color=colors[cls], linestyle='--', linewidth=1.5)
    plt.title(f"Confidence Distribution by TRUE Label\nModel: {model_name}", fontsize=14)
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Plot KDE by PREDICTED labels (right)
    plt.subplot(1, 2, 2)
    for cls in [0, 1]:
        sns.kdeplot(
            data=df[df[pred_label_col] == cls],
            x=confidence_score_col,
            fill=True,
            common_norm=False,
            alpha=0.5,
            label=f"Class {cls} ({'nDNA' if cls == 0 else 'mtDNA'})",
            color=colors[cls],
        )
        mean_val = df[df[pred_label_col] == cls][confidence_score_col].mean()
        plt.axvline(mean_val, color=colors[cls], linestyle='--', linewidth=1.5)
    plt.title(f"Confidence Distribution by PREDICTED Label\nModel: {model_name}", fontsize=14)
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(score_dir, f"score_distribution_true_vs_pred_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")


# Base directory for results
base_dir = Path("/home/saram/PhD/genomit_PMD/saved_results")

# File paths dictionary for survey answers and gold data
file_paths = {
    "gpt-4o": base_dir / "survey" / "survey_answer_gpt-4o.csv",
    "gpt-4o-mini": base_dir / "survey" / "survey_answer_gpt-4o-mini.csv",
    "gpt-3.5-turbo": base_dir / "survey" / "survey_answer_gpt-3.5-turbo.csv",
    "sauerkrautlm-gemma-2-9b-it-i1": base_dir / "survey" / "survey_answer_sauerkrautlm-gemma-2-9b-it-i1.csv",
    "phi-4-mini-reasoning": base_dir / "survey" / "survey_answer_phi-4-mini-reasoning.csv",
    "deepseek-r1-distill-qwen-7b": base_dir / "survey" / "survey_answer_deepseek-r1-distill-qwen-7b.csv",
    "gold": base_dir / "df" / "df_symp.csv",
}

# Output directory for saving results
output_dir = base_dir / "survey"
output_dir.mkdir(parents=True, exist_ok=True)

# Load gold standard data
gold_df = pd.read_csv(file_paths["gold"])
gold_df.rename(columns={"subjid": "ID"}, inplace=True)
gold_df = gold_df[["ID", "gendna_type"]]

# Select models to include (uncomment or comment out as needed)
selected_models = {
    "gpt-4o": file_paths["gpt-4o"],
    "gpt-4o-mini": file_paths["gpt-4o-mini"],
    "gpt-3.5-turbo": file_paths["gpt-3.5-turbo"],
    # "sauerkrautlm-gemma-2-9b-it-i1": file_paths["sauerkrautlm-gemma-2-9b-it-i1"],
    # "phi-4-mini-reasoning": file_paths["phi-4-mini-reasoning"],
    # "deepseek-r1-distill-qwen-7b": file_paths["deepseek-r1-distill-qwen-7b"],
}

# Initialize a list to collect metrics for all models
metrics_list = []

# Iterate over each model
for model_name, model_path in selected_models.items():
    print(f"Evaluating model: {model_name}")
    print (f"Model path: {model_path}")
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
    print(f"Confusion matrix plot saved for {model_name}")

    try:
        plot_confidence_distribution_by_class_side_by_side(
        df=merged_df,
        true_label_col="gendna_type",
        pred_label_col="mutation",
        confidence_score_col="score",
        model_name=model_name,
        output_dir=output_dir,
        )
    except IndexError as e:
        print(f"Skipping confidence distribution plot for {model_name} due to IndexError: {e}")


        # Analyze misclassifications
    clinical_diag_path = "/home/saram/PhD/genomit_PMD/saved_results/df/df_Global_preprocessed.csv"
    analyze_misclassifications(
        merged_df=merged_df,
        output_dir=output_dir,
        model_name=model_name,
        clindiag_mapping=CLINDIAG_MAPPING,
        clinical_diag_path=clinical_diag_path,
    )


# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Save metrics to CSV
metrics_df.to_csv(os.path.join(output_dir, "performance_comparison.csv"), index=False)

print("Performance evaluation completed. Results saved to the specified directory.")
# Print the final metrics DataFrame
print(metrics_df)


