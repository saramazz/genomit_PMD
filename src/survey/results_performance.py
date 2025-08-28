import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.cm as cm
import numpy as np

# --- Create folders if they do not exist ---
hist_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/hist"
cm_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/CM"
os.makedirs(hist_folder, exist_ok=True)
os.makedirs(cm_folder, exist_ok=True)


# --- Paths ---
input_file = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/results_summary_all_files_with_registry.json"
output_file = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/patient_level_results.csv"
metrics_file = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/performance_metrics.json"

# --- Load data ---
with open(input_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

#print the number of unique clinicians and patients
print(f"Loaded {len(df)} survey entries from {input_file}")
print(f"Unique clinicians: {df['clinician_id'].nunique()}")
print(f"Unique patients: {df['id_registry'].nunique()}")


# --- Encode labels ---
label_map = {"mtDNA": 1, "nDNA": 0}
df["gold_label_enc"] = df["gold_label"].map(label_map)
df["ml_label_enc"] = df["ml_label"].map(label_map)
df["llm_label_enc"] = df["LLM_prediction"].map(label_map)
df["clinician_answer_enc"] = df["answer_clinician"].map(label_map)

# Extract role and expertise from clinician_id
df["role"] = df["clinician_id"].apply(lambda x: "adult" if "adulto" in x else ("pediatric" if "pediatrico" in x else "other"))
df["expertise"] = df["clinician_id"].apply(lambda x: "expert" if "esperto" in x and "non esperto" not in x else "non_expert")

# --- Aggregate per patient ---
patient_results = []

for pid, group in df.groupby("id_registry"):
    gold = group["gold_label_enc"].iloc[0]
    ml = group["ml_label_enc"].iloc[0]
    llm = group["llm_label_enc"].iloc[0]
    aao = group["aao"].iloc[0]  # "adu" or "ped"

    # Average predictions
    overall_avg = round(group["clinician_answer_enc"].mean())

    # Experts vs non-experts
    expert_group = group[group["expertise"] == "expert"]
    nonexpert_group = group[group["expertise"] == "non_expert"]

    expert_avg = round(expert_group["clinician_answer_enc"].mean()) if not expert_group.empty else None
    nonexpert_avg = round(nonexpert_group["clinician_answer_enc"].mean()) if not nonexpert_group.empty else None

    # Adult vs pediatric clinicians (overall)
    adult_group = group[group["role"] == "adult"]
    ped_group = group[group["role"] == "pediatric"]

    adult_avg = round(adult_group["clinician_answer_enc"].mean()) if not adult_group.empty else None
    ped_avg = round(ped_group["clinician_answer_enc"].mean()) if not ped_group.empty else None

    # Adult on adult / pediatric
    adult_on_adult = round(adult_group[adult_group["aao"] == "adu"]["clinician_answer_enc"].mean()) if aao == "adu" and not adult_group.empty else None
    adult_on_ped   = round(adult_group[adult_group["aao"] == "ped"]["clinician_answer_enc"].mean()) if aao == "ped" and not adult_group.empty else None

    # Pediatric on pediatric / adult
    ped_on_ped = round(ped_group[ped_group["aao"] == "ped"]["clinician_answer_enc"].mean()) if aao == "ped" and not ped_group.empty else None
    ped_on_adult = round(ped_group[ped_group["aao"] == "adu"]["clinician_answer_enc"].mean()) if aao == "adu" and not ped_group.empty else None

    patient_results.append({
        "id_registry": pid,
        "gold_label": gold,
        "ml_prediction": ml,
        "LLM_prediction": llm,
        "avg_clinician_prediction": overall_avg,
        "avg_expert_prediction": expert_avg,
        "avg_nonexpert_prediction": nonexpert_avg,
        "avg_adult_prediction": adult_avg,
        "avg_pediatric_prediction": ped_avg,
        "adult_on_adult": adult_on_adult,
        "adult_on_pediatric": adult_on_ped,
        "pediatric_on_pediatric": ped_on_ped,
        "pediatric_on_adult": ped_on_adult,
        "aao": aao
    })

results_df = pd.DataFrame(patient_results)
results_df.to_csv(output_file, index=False)
print(f"Saved per-patient predictions to {output_file}")

# --- Compute metrics ---
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

metrics = {"ML": compute_metrics(results_df["gold_label"], results_df["ml_prediction"]), "LLM": compute_metrics(results_df["gold_label"], results_df["LLM_prediction"])}


for col in ["avg_clinician_prediction", "avg_expert_prediction", "avg_nonexpert_prediction",
            "avg_adult_prediction", "avg_pediatric_prediction",
            "adult_on_adult", "adult_on_pediatric",
            "pediatric_on_pediatric", "pediatric_on_adult"]:
    valid = results_df[results_df[col].notna()]
    if not valid.empty:
        metrics[col] = compute_metrics(valid["gold_label"], valid[col])

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved metrics to {metrics_file}")



# --- Function to plot bar chart ---
def plot_bar(metrics_dict, metric_name="accuracy", save_path=None):
    plt.figure(figsize=(12,7))

    # Sort metrics
    vals = {k: v[metric_name] for k,v in metrics_dict.items()}
    sorted_vals = dict(sorted(vals.items(), key=lambda x: x[1], reverse=True))

    # Color gradient
    colors = cm.Blues(np.linspace(0.4, 0.9, len(sorted_vals)))

    bars = plt.bar(sorted_vals.keys(), sorted_vals.values(), color=colors)

    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Comparison of ML and Clinician Groups vs Gold Label ({metric_name})")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    #plt.show()

# --- Plot Accuracy ---
plot_bar(metrics, metric_name="accuracy", save_path=os.path.join(hist_folder, "accuracy_comparison.svg"))

# --- Plot F1 ---
plot_bar(metrics, metric_name="f1", save_path=os.path.join(hist_folder, "f1_comparison.svg"))

# --- Generate confusion matrices ---
for group, vals in metrics.items():
    # Get predictions for that group
    if group == "ML":
        y_pred = results_df["ml_prediction"]
        y_true = results_df["gold_label"]
    elif group == "LLM":
        y_pred = results_df["LLM_prediction"]
        y_true = results_df["gold_label"]
    else:
        if group not in results_df.columns:
            continue
        y_pred = results_df[group].dropna()
        y_true = results_df.loc[y_pred.index, "gold_label"]
        y_pred = y_pred.astype(int)
        y_true = y_true.astype(int)

    cmatrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cmatrix, display_labels=["nDNA","mtDNA"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {group}")
    plt.tight_layout()
    plt.savefig(os.path.join(cm_folder, f"CM_{group}.svg"), format="svg")
    plt.close()

from sklearn.metrics import precision_recall_fscore_support

# --- Compute weighted metrics ---
def compute_weighted_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = (y_true == y_pred).mean()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

metrics_weighted = {}

for group in ["ML", "LLM", "avg_clinician_prediction", "avg_expert_prediction", "avg_nonexpert_prediction",
            "avg_adult_prediction", "avg_pediatric_prediction",
            "adult_on_adult", "adult_on_pediatric",
            "pediatric_on_pediatric", "pediatric_on_adult"]:
    
    if group == "ML":
        y_pred = results_df["ml_prediction"]
        y_true = results_df["gold_label"]
    elif group == "LLM":
        y_pred = results_df["LLM_prediction"]
        y_true = results_df["gold_label"]
    elif group in results_df.columns:
        y_pred = results_df[group].dropna()
        y_true = results_df.loc[y_pred.index, "gold_label"]
    else:
        continue

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    metrics_weighted[group] = compute_weighted_metrics(y_true, y_pred)

# --- Plot weighted accuracy ---
def plot_weighted_bar(metrics_dict, metric_name="accuracy", save_path=None):
    plt.figure(figsize=(12,7))
    vals = {k: v[metric_name] for k,v in metrics_dict.items()}
    sorted_vals = dict(sorted(vals.items(), key=lambda x: x[1], reverse=True))
    colors = cm.Blues(np.linspace(0.4, 0.9, len(sorted_vals)))

    bars = plt.bar(sorted_vals.keys(), sorted_vals.values(), color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Weighted {metric_name.capitalize()} Comparison of ML and Clinician Groups")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    #plt.show()

# Save weighted accuracy and F1
plot_weighted_bar(metrics_weighted, metric_name="accuracy", save_path=os.path.join(hist_folder, "weighted_accuracy_comparison.svg"))
plot_weighted_bar(metrics_weighted, metric_name="f1", save_path=os.path.join(hist_folder, "weighted_f1_comparison.svg"))
