# --- System / OS ---
import os
import json

# --- Data handling ---
import pandas as pd
import numpy as np

# --- Metrics ---
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.metrics import roc_auc_score

# --- Plotting ---
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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


# --- Function to compute full set of metrics ---
def compute_full_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,      # Sensitivity
        "specificity": specificity,
        "f1": f1
    }
# --- Function to compute metrics with AUC ---
def compute_auc_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.0
    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity
    }


hist_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/hist"
cm_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/CM"
os.makedirs(hist_folder, exist_ok=True)
os.makedirs(cm_folder, exist_ok=True)

# --- Compute metrics for each group ---
metrics_full = {}
for group in [
    "ML", "LLM", "avg_clinician_prediction", "avg_expert_prediction",
    "avg_nonexpert_prediction", "avg_adult_prediction", "avg_pediatric_prediction",
    "adult_on_adult", "adult_on_pediatric",
    "pediatric_on_pediatric", "pediatric_on_adult"
]:
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
    metrics_full[group] = compute_full_metrics(y_true, y_pred)

# --- Add subgroup metrics (Adults / Pediatrics only) ---
adult_df = results_df[results_df["aao"] == "adu"]
metrics_full["ML_adult"] = compute_full_metrics(adult_df["gold_label"], adult_df["ml_prediction"])
metrics_full["LLM_adult"] = compute_full_metrics(adult_df["gold_label"], adult_df["LLM_prediction"])

ped_df = results_df[results_df["aao"] == "ped"]
metrics_full["ML_pediatric"] = compute_full_metrics(ped_df["gold_label"], ped_df["ml_prediction"])
metrics_full["LLM_pediatric"] = compute_full_metrics(ped_df["gold_label"], ped_df["LLM_prediction"])

# --- Improved function for grouped barplots with external legend ---
def plot_grouped_bar(metrics_dict, groups, title, save_path):
    metric_names = ["f1", "precision", "recall", "specificity"]
    x = np.arange(len(groups))
    width = 0.18  # bar width

    plt.figure(figsize=(20,6))

    # Color gradient for metrics
    colors = cm.Blues(np.linspace(0.4, 0.9, len(metric_names)))

    for i, metric in enumerate(metric_names):
        values = [metrics_dict[g][metric] for g in groups if g in metrics_dict]
        bars = plt.bar(
            x + i*width, values, width,
            label=metric.capitalize(),
            color=colors[i]
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

    # Formatting
    plt.xticks(x + width*(len(metric_names)-1)/2, groups, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0,1.05)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Place legend outside on the right
    plt.legend(
        title="Metric",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    plt.tight_layout(rect=[0, 0, 0.98, 1])  # leave space for legend
    plt.savefig(save_path, format="svg")
    plt.close()

# --- Panel A: All patients ---
groups_A = [
    "ML", "LLM", "avg_clinician_prediction",
    "avg_adult_prediction", "avg_pediatric_prediction",
    "avg_expert_prediction", "avg_nonexpert_prediction"
]
plot_grouped_bar(metrics_full, groups_A, "Panel A: All patients",
                 os.path.join(hist_folder, "panel_A.svg"))

# --- Panel B: Adult patients only ---
groups_B = ["ML_adult", "LLM_adult", "adult_on_adult", "pediatric_on_adult"]
plot_grouped_bar(metrics_full, groups_B, "Panel B: Adult patients",
                 os.path.join(hist_folder, "panel_B.svg"))

# --- Panel C: Pediatric patients only ---
groups_C = ["ML_pediatric", "LLM_pediatric", "adult_on_pediatric", "pediatric_on_pediatric"]
plot_grouped_bar(metrics_full, groups_C, "Panel C: Pediatric patients",
                 os.path.join(hist_folder, "panel_C.svg"))

# --- Generate confusion matrices for all groups ---
all_groups = groups_A + groups_B + groups_C
for group in all_groups:
    if group not in metrics_full:
        continue

    if group == "ML":
        y_pred = results_df["ml_prediction"]
        y_true = results_df["gold_label"]
    elif group == "LLM":
        y_pred = results_df["LLM_prediction"]
        y_true = results_df["gold_label"]
    elif group == "ML_adult":
        y_pred = adult_df["ml_prediction"]
        y_true = adult_df["gold_label"]
    elif group == "LLM_adult":
        y_pred = adult_df["LLM_prediction"]
        y_true = adult_df["gold_label"]
    elif group == "ML_pediatric":
        y_pred = ped_df["ml_prediction"]
        y_true = ped_df["gold_label"]
    elif group == "LLM_pediatric":
        y_pred = ped_df["LLM_prediction"]
        y_true = ped_df["gold_label"]
    else:
        y_pred = results_df[group].dropna()
        y_true = results_df.loc[y_pred.index, "gold_label"]

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)

    cmatrix = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cmatrix, display_labels=["nDNA","mtDNA"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {group}")
    plt.tight_layout()
    plt.savefig(os.path.join(cm_folder, f"CM_{group}.svg"), format="svg")
    plt.close()

def plot_grouped_panels(metrics_dict, panels, save_path):
    metric_names = ["f1", "precision", "recall", "specificity"]
    max_groups = max(len(groups) for _, groups in panels)
    width = 0.18

    fig, axes = plt.subplots(len(panels), 1, figsize=(22, 16), sharey=True)
    colors = cm.Blues(np.linspace(0.4, 0.9, len(metric_names)))

    for ax, (title, groups) in zip(axes, panels):
        x = np.arange(max_groups)
        for i, metric in enumerate(metric_names):
            values = [metrics_dict[g][metric] if g in metrics_dict else 0 for g in groups]
            values += [0] * (max_groups - len(groups))

            bars = ax.bar(
                x + i * width, values, width,
                label=metric.capitalize(),
                color=colors[i]
            )

            for j, (bar, val) in enumerate(zip(bars, values)):
                if j < len(groups) and val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold"
                    )

        labels = groups + [""] * (max_groups - len(groups))
        ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylabel("Score")

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, title="Metric",
        loc="center left", bbox_to_anchor=(0.8, 0.5),
        borderaxespad=0
    )

    # Adjust spacing for all panels and x-labels
    fig.subplots_adjust(right=0.78, hspace=0.5, bottom=0.15)
    fig.savefig(save_path, format="svg", bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()


# --- Example usage ---
panels = [
    ("All patients", [
        "ML", "LLM", "avg_clinician_prediction",
        "avg_adult_prediction", "avg_pediatric_prediction",
        "avg_expert_prediction", "avg_nonexpert_prediction"
    ]),
    ("Adult patients", [
        "ML_adult", "LLM_adult", "adult_on_adult", "pediatric_on_adult"
    ]),
    ("Pediatric patients", [
        "ML_pediatric", "LLM_pediatric", "pediatric_on_pediatric", "adult_on_pediatric"
    ])
]

plot_grouped_panels(metrics_full, panels, os.path.join(hist_folder, "panel_ABC.svg"))



### AUC Plots ###
# --- Recompute metrics_full with new function ---
metrics_full = {}
for group in [
    "ML", "LLM", "avg_clinician_prediction", "avg_expert_prediction",
    "avg_nonexpert_prediction", "avg_adult_prediction", "avg_pediatric_prediction",
    "adult_on_adult", "adult_on_pediatric",
    "pediatric_on_pediatric", "pediatric_on_adult"
]:
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
    metrics_full[group] = compute_auc_metrics(y_true, y_pred)

# --- Subgroups ---
metrics_full["ML_adult"] = compute_auc_metrics(adult_df["gold_label"], adult_df["ml_prediction"])
metrics_full["LLM_adult"] = compute_auc_metrics(adult_df["gold_label"], adult_df["LLM_prediction"])
metrics_full["ML_pediatric"] = compute_auc_metrics(ped_df["gold_label"], ped_df["ml_prediction"])
metrics_full["LLM_pediatric"] = compute_auc_metrics(ped_df["gold_label"], ped_df["LLM_prediction"])

# --- Update plotting functions ---
def plot_grouped_bar(metrics_dict, groups, title, save_path):
    metric_names = ["auc", "sensitivity", "specificity"]
    x = np.arange(len(groups))
    width = 0.22

    plt.figure(figsize=(20, 6))
    colors = cm.Blues(np.linspace(0.4, 0.9, len(metric_names)))

    for i, metric in enumerate(metric_names):
        values = [metrics_dict[g][metric] for g in groups if g in metrics_dict]
        bars = plt.bar(
            x + i * width, values, width,
            label=metric.capitalize(),
            color=colors[i]
        )
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

    plt.xticks(x + width, groups, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Metric", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    plt.savefig(save_path, format="svg")
    plt.close()

def plot_grouped_panels(metrics_dict, panels, save_path):
    metric_names = ["auc", "sensitivity", "specificity"]
    max_groups = max(len(groups) for _, groups in panels)
    width = 0.22

    fig, axes = plt.subplots(len(panels), 1, figsize=(22, 16), sharey=True)
    colors = cm.Blues(np.linspace(0.4, 0.9, len(metric_names)))

    for ax, (title, groups) in zip(axes, panels):
        x = np.arange(max_groups)
        for i, metric in enumerate(metric_names):
            values = [metrics_dict[g][metric] if g in metrics_dict else 0 for g in groups]
            values += [0] * (max_groups - len(groups))

            bars = ax.bar(x + i * width, values, width,
                          label=metric.capitalize(), color=colors[i])
            for j, (bar, val) in enumerate(zip(bars, values)):
                if j < len(groups) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}",
                            ha="center", va="bottom",
                            fontsize=8, fontweight="bold")

        labels = groups + [""] * (max_groups - len(groups))
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, title="Metric",
                        loc="center left", bbox_to_anchor=(0.8, 0.5),
                        borderaxespad=0)

    fig.subplots_adjust(right=0.78, hspace=0.5, bottom=0.15)
    fig.savefig(save_path, format="svg", bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()


# --- Panel A: All patients ---
groups_A = [
    "ML", "LLM", "avg_clinician_prediction",
    "avg_adult_prediction", "avg_pediatric_prediction",
    "avg_expert_prediction", "avg_nonexpert_prediction"
]
plot_grouped_bar(metrics_full, groups_A, "Panel A: All patients",
                 os.path.join(hist_folder, "panel_A_auc.svg"))

# --- Panel B: Adult patients only ---
groups_B = ["ML_adult", "LLM_adult", "adult_on_adult", "pediatric_on_adult"]
plot_grouped_bar(metrics_full, groups_B, "Panel B: Adult patients",
                 os.path.join(hist_folder, "panel_B_auc.svg"))

# --- Panel C: Pediatric patients only ---
groups_C = ["ML_pediatric", "LLM_pediatric", "adult_on_pediatric", "pediatric_on_pediatric"]
plot_grouped_bar(metrics_full, groups_C, "Panel C: Pediatric patients",
                 os.path.join(hist_folder, "panel_C_auc.svg"))

# --- Example usage ---
panels = [
    ("All patients", [
        "ML", "LLM", "avg_clinician_prediction",
        "avg_adult_prediction", "avg_pediatric_prediction",
        "avg_expert_prediction", "avg_nonexpert_prediction"
    ]),
    ("Adult patients", [
        "ML_adult", "LLM_adult", "adult_on_adult", "pediatric_on_adult"
    ]),
    ("Pediatric patients", [
        "ML_pediatric", "LLM_pediatric", "pediatric_on_pediatric", "adult_on_pediatric"
    ])
]

plot_grouped_panels(metrics_full, panels, os.path.join(hist_folder, "panel_ABC_auc.svg"))

from sklearn.metrics import precision_recall_fscore_support

def compute_class_metrics(y_true, y_pred, labels=["mtDNA", "nDNA"]):
    """
    Calcola precision, recall e F1 per ciascuna classe e per la media pesata.
    """
    metrics = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )

    for i, label in enumerate(labels):
        metrics[label] = {
            "precision": round(precision[i], 2),
            "recall": round(recall[i], 2),
            "f1": round(f1[i], 2),
            "support": support[i]
        }

    # Weighted average
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted"
    )
    metrics["weighted"] = {
        "precision": round(precision_w, 2),
        "recall": round(recall_w, 2),
        "f1": round(f1_w, 2)
    }

    return metrics

#use the function to compute metrics for clinicians overall
overall_metrics = compute_class_metrics(df["gold_label"], df["answer_clinician"])

print("Overall Clinician Metrics:")
print(overall_metrics)

#print the overall_metrics
#save the overall_metrics to a json file
with open(metrics_file, "w") as f:
    json.dump(overall_metrics, f, indent=2)
# --- Update plotting functions for AUC metrics ---