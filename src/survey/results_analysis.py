import os
import json
import pandas as pd
import re

folder_path = "/home/sam/sara_git/genomit_PMD/saved_results/survey_answers"
results = []

xlsx_files = [f for f in os.listdir(folder_path) if "Classificazione" in f and f.endswith(".xlsx")]
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

    print(f"Extracted clinician ID: '{clinician_id}'")

    xlsx_path = os.path.join(folder_path, xlsx_file)
    csv_path = os.path.splitext(xlsx_path)[0] + ".csv"

    if not os.path.exists(csv_path):
        print(f"Converting {xlsx_file} to CSV...")
        df_xlsx = pd.read_excel(xlsx_path, sheet_name=0)
        df_xlsx.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV saved as: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna('')

    question_rows = {}
    for idx, row in df.iterrows():
        first_col = row.iloc[0].strip()
        for patient_id in range(5, 45):
            if first_col.startswith(f"Q{patient_id}."):
                question_rows[patient_id] = idx
                break

    for patient_id in range(5, 45):
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
                perc = float(second_col.replace('%', '').strip())
            except ValueError:
                perc = None

            if first_col == "DNA nucleare" and perc is not None:
                dna_nuclear_percent = perc
                print(f"Patient {patient_id} (file {xlsx_file}): DNA nucleare = {dna_nuclear_percent}%")
            elif first_col == "DNA mitocondriale" and perc is not None:
                dna_mitochondrial_percent = perc
                print(f"Patient {patient_id} (file {xlsx_file}): DNA mitocondriale = {dna_mitochondrial_percent}%")

        answer = None
        if dna_nuclear_percent == 1.0:
            answer = "nDNA"
        elif dna_mitochondrial_percent == 1.0:
            answer = "mtDNA"

        if answer is None:
            print(f"Patient {patient_id} (file {xlsx_file}): No 100% answer found.")
        else:
            print(f"Patient {patient_id} (file {xlsx_file}): Answer = {answer}")
            results.append({
                #"filename": xlsx_file,
                "id_patient": patient_id,
                "answer_clinician": answer,
                "clinician_id": clinician_id
            })

output_file = os.path.join(folder_path, "results_summary_all_files.json")
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)

print(f"\n✅ Done. All results saved to {output_file}")
