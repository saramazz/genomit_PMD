import os
import json
import pandas as pd
import re

# folder_path = "/home/sam/sara_git/genomit_PMD/saved_results/survey_answers"
folder_path = (
    "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers/export_website"
)
output_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers"
results = []

xlsx_files = [
    f for f in os.listdir(folder_path) if "Classificazione" in f and f.endswith(".xlsx")
]
if not xlsx_files:
    print("No matching files found.")
    exit()

# Fixed prefix to find clinician ID after it
fixed_prefix = "Classificazione nDNA - mtDNA nelle malattie mitocondriali il neurologo a confronto con lAI"

for xlsx_file in xlsx_files:
    print(f"Processing file: {xlsx_file}")

    # Extract clinician ID as anything after the fixed prefix in filename (before .xlsx)
    clinician_id_match = re.search(re.escape(fixed_prefix) + r"(.*)\.xlsx$", xlsx_file)
    if clinician_id_match:
        clinician_id = clinician_id_match.group(1).strip("_ ").strip()
    else:
        clinician_id = "unknown"

    # print(f"Extracted clinician ID: '{clinician_id}'")

    xlsx_path = os.path.join(folder_path, xlsx_file)
    csv_path = os.path.splitext(xlsx_path)[0] + ".csv"

    if not os.path.exists(csv_path):
        print(f"Converting {xlsx_file} to CSV...")
        df_xlsx = pd.read_excel(xlsx_path, sheet_name=0)
        df_xlsx.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV saved as: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    question_rows = {}
    for idx, row in df.iterrows():
        first_col = row.iloc[0].strip()
        for patient_id in range(5, 45):
            if first_col.startswith(f"Q{patient_id}."):
                question_rows[patient_id] = idx
                break

    for patient_id in range(5, 25):
        if patient_id not in question_rows:
            print(f"Question Q{patient_id} not found in file {xlsx_file}.")
            continue

        q_idx = question_rows[patient_id]

        dna_nuclear_percent = None
        dna_mitochondrial_percent = None

        for offset in range(1, 11):
            i = q_idx + offset
            if i >= len(df):
                break
            first_col = df.iloc[i, 0].strip()
            second_col = df.iloc[i, 1].strip()

            try:
                perc = float(second_col.replace("%", "").strip())
            except ValueError:
                perc = None

            if first_col == "DNA nucleare" and perc is not None:
                dna_nuclear_percent = perc

            elif first_col == "DNA mitocondriale" and perc is not None:
                dna_mitochondrial_percent = perc

        answer = None
        if dna_nuclear_percent == 1.0:
            answer = "nDNA"
        elif dna_mitochondrial_percent == 1.0:
            answer = "mtDNA"

        if answer is None:
            print(f"Patient {patient_id} (file {xlsx_file}): No 100% answer found.")
        else:
            # print(f"Patient {patient_id} (file {xlsx_file}): Answer = {answer}")
            results.append(
                {
                    # "filename": xlsx_file,
                    "id_answer": patient_id,
                    "answer_clinician": answer,
                    "clinician_id": clinician_id,
                }
            )


output_file = os.path.join(output_folder, "results_summary_all_files.json")
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)

print(f"\n✅ Done. All results saved to {output_file}")


import os
import json
import pandas as pd

# Paths
output_folder = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answers"
survey_answer_file = os.path.join(output_folder, "results_summary_all_files.json")
id_pat_mapping_file = os.path.join(output_folder, "id_pat_mapping.csv")
id_pati_clin_file = os.path.join(output_folder, "id_pati_clin.csv")
ml_predictions_file = "/home/saram/PhD/genomit_PMD/saved_results/classifiers_results/experiments_all_models_red/df_test_best_fp_fn.csv"


# --- Load files ---
with open(survey_answer_file, "r") as f:
    survey_data = json.load(f)

# For testing: limit to first 10 entries
# survey_data = survey_data[:10]

print(f"Loaded {len(survey_data)} survey entries")
print(f"Loading ID mapping from {id_pat_mapping_file} and {id_pati_clin_file}")

id_pat_mapping = pd.read_csv(id_pat_mapping_file, sep=",")
id_pati_clin = pd.read_csv(id_pati_clin_file, sep=",")
ml_predictions = pd.read_csv(ml_predictions_file, sep=",")


# --- Build mapping from "Paziente N" to {id_registry, gold_label, aao} ---
paziente_to_info = {
    row["id_survey"]: {
        "id_registry": row["id_registry"],
        "gold_label": row["gold_label"],
        "aao": row.get("aao", None),  # aggiungi AAO se disponibile
    }
    for _, row in id_pat_mapping.iterrows()
}

# --- Build mapping id_registry -> ml_label ---
registry_to_ml = dict(zip(ml_predictions["Subject id"], ml_predictions["predictions"]))
# substitute prediction 1 with mtDNA and 0 with nDNA
registry_to_ml = {k: ("mtDNA" if v == 1 else "nDNA") for k, v in registry_to_ml.items()}


# --- Helper: normalize clinician names (Italian → file format) ---
import re


def normalize_clinician_name(clinician_str: str) -> str:
    # Example inputs:
    # "adulto esperto 18"
    # "pediatrico non esperto 45"
    # "adulto non esperto 28"

    # Extract trailing number
    match = re.search(r"(\d+)$", clinician_str)
    if not match:
        return clinician_str
    idx = match.group(1)

    text = clinician_str.lower().replace(idx, "").strip()  # remove number part
    # Possible patterns:
    # "adulto esperto"
    # "adulto non esperto"
    # "pediatrico esperto"
    # "pediatrico non esperto"

    role_en = None
    exp_en = None

    if text.startswith("adulto"):
        role_en = "adult"
        if "non esperto" in text:
            exp_en = "Nonexpert"
        else:
            exp_en = "Expert"
    elif text.startswith("pediatrico"):
        role_en = "pediatric"
        if "non esperto" in text:
            exp_en = "Nonexpert"
        else:
            exp_en = "Expert"
    else:
        role_en = text.replace(" ", "_").capitalize()
        exp_en = "Expert"

    return f"Clinician_{idx}_{role_en}_{exp_en}_"


# --- Build mapping clinician -> list of patients ---
clinician_to_patients = {
    col: id_pati_clin[col].dropna().tolist() for col in id_pati_clin.columns
}
# --- Enrich survey data with id_registry, gold_label, ml_label ---
updated_data = []
clinician_counters = {}

for entry in survey_data:
    clinician_it = entry["clinician_id"]

    clinician_norm = normalize_clinician_name(clinician_it)

    if clinician_norm not in clinician_to_patients:
        print(
            f"Warning: clinician {clinician_it} → {clinician_norm} not found in mapping"
        )
        updated_data.append(entry)
        continue

    idx = clinician_counters.get(clinician_norm, 0)
    patients_list = clinician_to_patients[clinician_norm]

    if idx >= len(patients_list):
        print(f"Warning: more answers than patients for {clinician_it}")
        updated_data.append(entry)
        continue

    paziente_label = patients_list[idx]  # e.g. "Paziente 49"
    clinician_counters[clinician_norm] = idx + 1

    # Lookup registry + gold_label + aao
    info = paziente_to_info.get(paziente_label, {})
    id_registry = info.get("id_registry")
    gold_label = info.get("gold_label")
    aao = info.get("aao")

    # Lookup ml_label using id_registry
    ml_label = registry_to_ml.get(id_registry)

    enriched_entry = entry.copy()
    enriched_entry["id_registry"] = id_registry
    enriched_entry["gold_label"] = gold_label
    enriched_entry["ml_label"] = ml_label
    enriched_entry["aao"] = aao

    updated_data.append(enriched_entry)

# --- Load LLM predictions ---
deepseek_file = "/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_deepseek-r1-distill-qwen-7b.csv"
llm_df = pd.read_csv(deepseek_file, sep=",")  # assuming comma separator

# Build mapping: id_registry -> LLM_prediction
llm_mapping = dict(zip(llm_df["ID"], llm_df["mutation"]))
# substitute 1.0 with mtDNA and 0.0 with nDNA
llm_mapping = {k: ("mtDNA" if v == 1.0 else "nDNA") for k, v in llm_mapping.items()}

# --- Enrich survey_data with LLM_prediction ---
for entry in updated_data:
    id_registry = entry.get("id_registry")
    if id_registry:
        entry["LLM_prediction"] = llm_mapping.get(id_registry, None)
    else:
        entry["LLM_prediction"] = None

# --- Save updated JSON ---
output_file_llm = os.path.join(
    output_folder, "results_summary_all_files_with_registry.json"
)
with open(output_file_llm, "w") as f:
    json.dump(updated_data, f, indent=4, ensure_ascii=False)

print(f"Updated file with LLM predictions saved to: {output_file_llm}")
