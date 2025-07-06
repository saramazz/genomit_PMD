import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from threading import Lock
import json
import re
from datetime import datetime
from openai import OpenAI


def extract_class_score(response_text):
    """
    Extract DNA class and score from JSON in response_text.
    Returns (dna_class, score), np.nan if missing or invalid.
    """
    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")
        json_str = match.group(0)
        result = json.loads(json_str)

        dna_val = result.get("DNA", None)
        dna_class = int(dna_val) if str(dna_val) in ["0", "1"] else np.nan

        conf_val = result.get("score", None)
        score = float(conf_val) if conf_val is not None else np.nan
        if not (0 <= score <= 1):
            score = np.nan

        return dna_class, score
    except Exception as e:
        logger.error(f"Failed to parse response: {e}\nResponse was:\n{response_text}")
        return np.nan, np.nan


def call_gpt_api(description, model, max_tokens=2000, temperature=0.2):
    messages = [
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
                "- Set 'score' as follows:\n"
                "  • If you are almost certain the mutation is in mtDNA, return a score near 1.0 (e.g., 0.95 or 0.99).\n"
                "  • If you are almost certain the mutation is in nDNA, return a score near 0.0 (e.g., 0.05 or 0.10).\n"
                "  • If you are unsure or the evidence is ambiguous, use values in the middle (e.g., 0.5).\n"
                "- Do not use fixed or default values like 0.85 for all cases.\n"
                "- Make sure your score reflects the actual degree of certainty based only on the information provided."
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
        - The 'DNA' value is 1 if mtDNA is more likely, 0 if nDNA is more likely.
        - The 'score' expresses the probability that the mutation is in mtDNA. It should vary depending on how confident you are.
        - If you are unsure, return an intermediate score such as 0.5.
        - Avoid always returning 0.85. Vary the score appropriately based on the evidence.
        """,
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content.strip()
        dna_class, score = extract_class_score(content)
        return dna_class, score, content
    except Exception as e:
        logger.error(f"API exception: {e}")
        return np.nan, np.nan, ""


def append_result(result_row):
    with save_lock:
        pd.DataFrame([result_row]).to_csv(
            saving_path, mode="a", header=False, index=False
        )


def analyze_row(row, model, max_tokens, temperature):
    global processed_counter
    description = str(row["description"])

    dna_class, score, full_response = call_gpt_api(
        description, model, max_tokens, temperature
    )

    result_row = {
        "ID": row["ID"],
        "mutation": dna_class,
        "score": score,
        "reasoning": full_response,
    }

    logger.info(f"Processed ID {row['ID']}: {result_row}")
    append_result(result_row)

    with counter_lock:
        processed_counter += 1
        percent = (processed_counter / total_patients) * 100
        logger.info(
            f"Progress: {processed_counter}/{total_patients} patients processed ({percent:.2f}%)"
        )


# Initialize OpenAI client
client = OpenAI(api_key="API")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock for thread safety
save_lock = Lock()

# Base configurations
# model = "gpt-3.5-turbo"  # or "gpt-4o"
# model = "gpt-4o-mini"  # or "gpt-4o"
model = "gpt-4o"  # or "gpt-4o"
max_tokens = 2000
temperature = 0.2

# File configurations
file_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/Description4Survey.csv"
saving_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_{model}.csv"
)

# Load input data
data = pd.read_csv(file_path)
total_patients = len(data)
logger.info(f"Total patients to process: {total_patients}")

# Uncomment to test only a subset
# data = data.iloc[:2]


# Check and filter already processed entries
if os.path.exists(saving_path) and os.path.getsize(saving_path) > 0:
    processed_ids = pd.read_csv(saving_path)["ID"]
    data = data[~data["ID"].isin(processed_ids)]
    already_processed = len(processed_ids)
else:
    already_processed = 0

processed_counter = already_processed
counter_lock = Lock()

# Create output file with all expected columns if it doesn't exist
if not os.path.exists(saving_path):
    pd.DataFrame(columns=["ID", "mutation", "score", "reasoning"]).to_csv(
        saving_path, index=False
    )
# --------------------
# Start processing timer
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

params_log_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/params_log_{model}.txt"
)
with open(params_log_path, "w") as f:
    f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
    f.write(f"Input File: {file_path}\n")
    f.write(f"Output File: {saving_path}\n")
    f.write(f"Model Used: {model}\n")
logger.info(f"Parameters logged to: {params_log_path}")
