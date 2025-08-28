import pandas as pd
import json
import os

# --- Paths ---
json_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/results_summary_all_files_with_registry.json"
csv_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_deepseek-r1-distill-qwen-7b.csv"
save_dir = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/llm_answers_and_gold"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "survey_answer_deepseek-r1-distill-qwen-7b_with_gold.csv")

# --- Load JSON file ---
with open(json_path, "r") as f:
    data = json.load(f)

# Create mapping: id_registry -> gold_label
id_to_gold = {entry["id_registry"]: entry["gold_label"] for entry in data}

# --- Load CSV file ---
df = pd.read_csv(csv_path)

# --- Map gold labels ---
df["gold_label"] = df["ID"].map(id_to_gold)

# --- Add numeric gold label ---
df["gold_label_num"] = df["gold_label"].apply(lambda x: 1 if x == "mtDNA" else 0)

# --- Save new file ---
df.to_csv(save_path, index=False)

print(f"✅ File saved with gold labels and numeric column at: {save_path}")
