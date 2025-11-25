# Standard library imports
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

# Third-party imports
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# Configuration
# base_url = "http://192.168.2.5:1234"  # "http://10.7.11.166:1234"
base_url = "http://192.168.1.8:1234"


# --------------------
# Set model name
# --------------------
model = "phi-4-mini-reasoning"
# model = "deepseek-r1-distill-qwen-7b"
# model = "sauerkrautlm-gemma-2-9b-it-i1" #COMPLETE
# model = "qwen2.5-7b-instruct" #COMPLETE

max_tokens = 3000
temperature = 0.2

# File paths
file_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/Description4Survey.csv"
saving_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_{model}_new.csv"
)
params_log_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/params_log_{model}_new.txt"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe counters and file locking
save_lock = Lock()
counter_lock = Lock()
processed_counter = 0


# Extract JSON fields from response
def extract_class_and_confidence(response_text):
    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        json_str = match.group(0)
        result = json.loads(json_str)

        dna_val = result.get("DNA", None)
        dna_class = int(dna_val) if str(dna_val) in ["0", "1"] else np.nan

        conf_val = result.get("score", None)
        confidence = float(conf_val) if conf_val is not None else np.nan
        if not (0 <= confidence <= 1):
            confidence = np.nan

        return dna_class, confidence
    except Exception as e:
        logger.error(
            f"Failed to extract DNA/confidence: {e}\nResponse was:\n{response_text}"
        )
        return np.nan, np.nan


session = requests.Session()
retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)
session.mount("https://", adapter)


def call_api(description, model, max_tokens=3000, temperature=0.2):
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are assisting with the classification of genetic mutations in patients with primary mitochondrial disorders. "
                    "Based on the patient's clinical history, determine whether the mutation is more likely to be located in mitochondrial DNA (mtDNA) or nuclear DNA (nDNA). "
                    "You must return two values:\n"
                    "1. A binary prediction ('DNA') indicating whether mtDNA (1) or nDNA (0) is more likely.\n"
                    "2. A confidence score ('score') between 0 and 1 representing the probability that the mutation is in mtDNA.\n\n"
                    "Instructions:\n"
                    "- Set 'DNA' to 1 if mtDNA is more likely, 0 if nDNA is more likely.\n"
                    "- Set 'score' accordingly:\n"
                    "  • Use values close to 1.0 (e.g., 0.95) if you are almost certain it is mtDNA.\n"
                    "  • Use values close to 0.0 (e.g., 0.05) if you are almost certain it is nDNA.\n"
                    "  • Use intermediate values (e.g., 0.5) if uncertain or ambiguous.\n"
                    "- Avoid defaulting to a fixed score like 0.85. Your score should reflect true confidence based on the provided information."
                ),
            },
            {
                "role": "user",
                "content": f"""Here is the patient's clinical history:
        {description}

        Please respond with a JSON object in the following format:
        {{
        "DNA": 0 or 1,
        "score": float between 0 and 1
        }}

        Make sure:
        - 'DNA' is 1 for mtDNA, 0 for nDNA.
        - 'score' reflects the probability that the mutation is in mtDNA.
        - If unsure, provide an intermediate confidence score.""",
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = session.post(url, headers=headers, json=payload, timeout=500)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        dna_class, confidence = extract_class_and_confidence(content)
        return dna_class, confidence, content
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return np.nan, np.nan, ""


# Append result to output CSV safely
def append_result(result_row):
    with save_lock:
        pd.DataFrame([result_row]).to_csv(
            saving_path, mode="a", header=False, index=False
        )


# Process a single patient description
def analyze_row(row, model, max_tokens, temperature):
    global processed_counter
    description = str(row["description"])

    dna_class, score, full_response = call_api(
        description, model, max_tokens, temperature
    )

    result_row = {
        "ID": row["ID"],
        "mutation": dna_class,
        "score": score,
        "reasoning": full_response,
    }

    logger.info(
        f"Processed ID {row['ID']}: {result_row['ID']} - result: {result_row['mutation']} with score {result_row['score']}"
    )
    append_result(result_row)

    with counter_lock:
        processed_counter += 1
        percent = (processed_counter / total_patients) * 100
        logger.info(
            f"Progress: {processed_counter}/{total_patients} patients processed ({percent:.2f}%)"
        )


# Main execution
if __name__ == "__main__":
    data = pd.read_csv(file_path)

    # decrease for testing
    # data = data.head(2)  # For testing purposes, limit to 10

    total_patients = len(data)
    logger.info(f"Total patients to process: {total_patients}")

    # model
    logger.info(f"Using model: {model}")

    # Skip already processed patients
    if os.path.exists(saving_path) and os.path.getsize(saving_path) > 0:
        df_partial = pd.read_csv(saving_path)

        # Identify rows where all columns except "ID" are NaN
        empty_rows_mask = df_partial.drop(columns=["ID"]).isnull().all(axis=1)
        num_empty_rows = empty_rows_mask.sum()

        if num_empty_rows > 0:
            # print dimensions of df_partial before cleaning
            logger.info(f"Dimensions of df_partial before cleaning: {df_partial.shape}")
            logger.warning(
                f"Found {num_empty_rows} rows with all NaN values (except ID) in {saving_path}. Removing them."
            )
            df_partial = df_partial[~empty_rows_mask]
            # Overwrite the CSV file without the empty rows
            df_partial.to_csv(saving_path, index=False)
            logger.info(f"Updated {saving_path} after removing empty rows.")
            # print dimensions of df_partial
            logger.info(f"Dimensions of df_partial after cleaning: {df_partial.shape}")
        else:
            logger.info("No empty rows (except ID) found in partial results.")

        # Extract IDs of successfully processed patients
        processed_ids = df_partial["ID"].tolist()

        # Remove already processed patients from the input data
        data = data[~data["ID"].isin(processed_ids)]

        processed_counter = len(processed_ids)
        logger.info(
            f"Skipping {len(processed_ids)} already processed patients. Remaining: {len(data)}"
        )
        # input("Press Enter to continue with the remaining patients...")
    else:
        processed_counter = 0
        pd.DataFrame(columns=["ID", "mutation", "score", "reasoning"]).to_csv(
            saving_path, index=False
        )
        logger.info(
            f"No previous results found. Starting fresh with {len(data)} patients."
        )

    # Start timing
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(analyze_row, row, model, max_tokens, temperature)
            for _, row in data.iterrows()
        ]
        for future in as_completed(futures):
            future.result()

    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete. Results saved to {saving_path}")

    # --------------------
    # Log Parameters Used
    # --------------------

    # Path for saving parameter log
    params_log_path = (
        f"/home/saram/PhD/genomit_PMD/saved_results/survey/params_log_{model}.txt"
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time if "start_time" in locals() else np.nan

    # Updated system and user prompt templates
    sys_content = (
        "You are assisting with the classification of genetic mutations in patients with primary mitochondrial disorders. "
        "Based on the patient's clinical history, determine whether the mutation is more likely to be located in mitochondrial DNA (mtDNA) or nuclear DNA (nDNA). "
        "You must return two values:\n"
        "1. A binary prediction ('DNA') indicating whether mtDNA (1) or nDNA (0) is more likely.\n"
        "2. A confidence score ('score') between 0 and 1 representing the probability that the mutation is in mtDNA.\n\n"
        "Instructions:\n"
        "- Set 'DNA' to 1 if mtDNA is more likely, 0 if nDNA is more likely.\n"
        "- Set 'score' as follows:\n"
        "  • If you are almost certain the mutation is in mtDNA, return a score near 1.0 (e.g., 0.95 or 0.99).\n"
        "  • If you are almost certain the mutation is in nDNA, return a score near 0.0 (e.g., 0.05 or 0.10).\n"
        "  • If you are unsure or the evidence is ambiguous, use values in the middle (e.g., 0.5).\n"
        "- Do not use fixed or default values like 0.85 for all cases.\n"
        "- Make sure your score reflects the actual degree of certainty based only on the information provided."
    )

    prompt_example = """Here is the patient's clinical history:
    {description}

    Please respond with a JSON object in the following format:
    {{
    "DNA": 0 or 1,
    "score": float between 0 and 1
    }}

    Make sure:
    - The 'DNA' value is 1 if mtDNA is more likely, 0 if nDNA is more likely.
    - The 'score' expresses the probability that the mutation is in mtDNA. It should vary depending on how confident you are.
    - If you are unsure, return an intermediate score such as 0.5.
    - Avoid always returning 0.85. Vary the score appropriately based on the evidence.
        """

    # Write all information to the log file
    with open(params_log_path, "w") as f:
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Input File: {file_path}\n")
        f.write(f"Output File: {saving_path}\n")
        f.write(f"Number of Records Processed: {len(data)}\n")
        f.write(f"Model Used: {model}\n")
        f.write(f"Max Tokens: {max_tokens}\n")
        f.write(f"System Prompt:\n{sys_content}\n")
        f.write(f"User Prompt Example:\n{prompt_example}\n")

    logger.info(f"Parameters logged to: {params_log_path}")
