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

# Initialize OpenAI client
client = OpenAI(
    api_key="API"
)
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock for thread safety
save_lock = Lock()

# Base configurations
model = "gpt-3.5-turbo"#"gpt-4o"
max_tokens = 1500
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

# Check and filter already processed entries
if os.path.exists(saving_path) and os.path.getsize(saving_path) > 0:
    processed_ids = pd.read_csv(saving_path)["ID"]
    data = data[~data["ID"].isin(processed_ids)]
    already_processed = len(processed_ids)
else:
    already_processed = 0

processed_counter = already_processed
counter_lock = Lock()


def extract_binary_answer(response_text):
    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")
        json_str = match.group(0)
        result = json.loads(json_str)
        val = result.get("DNA", None)
        if str(val) in ["0", "1"]:
            return int(val)
    except Exception as e:
        logger.error(f"Failed to parse response: {e}\nResponse was:\n{response_text}")
    return np.nan


def call_gpt_api(description, model, max_tokens=2000, temperature=0.2):
    messages = [
        {
            "role": "system",
            "content": (
                "You are assisting with the classification of genetic mutations in patients with primary mitochondrial disorders. "
                "Based on the clinical presentation, determine whether the mutation is more likely to be in mitochondrial DNA (mtDNA) or nuclear DNA (nDNA)."
            ),
        },
        {
            "role": "user",
            "content": f"""Here is the patient's clinical history:
    {description}
    Please respond with a JSON object in the following format:
    {{
    "DNA": 1  // if the mutation is more likely in mitochondrial DNA (mtDNA)
    }}
    or
    {{
    "DNA": 0  // if the mutation is more likely in nuclear DNA (nDNA)
    }}

    Instructions:
        - Base your reasoning only on the information provided in the description.
        - There are no strict criteria; make the best-informed guess.
        - Return only the JSON object. Do not include explanations or any other text.

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
        return extract_binary_answer(content)
    except Exception as e:
        logger.error(f"API exception: {e}")
        return np.nan


def append_result(result_row):
    with save_lock:
        pd.DataFrame([result_row]).to_csv(
            saving_path, mode="a", header=False, index=False
        )


def analyze_row(row, model, max_tokens, temperature):
    global processed_counter
    description = row["description"]
    result_row = {"ID": row["ID"]}
    mutation_result = call_gpt_api(description, model, max_tokens, temperature)
    result_row.update({"mutation": mutation_result})
    append_result(result_row)
    with counter_lock:
        processed_counter += 1
        percent = (processed_counter / total_patients) * 100
        logger.info(
            f"Progress: {processed_counter}/{total_patients} patients processed ({percent:.2f}%)"
        )


if not os.path.exists(saving_path):
    pd.DataFrame(columns=["ID", "mutation"]).to_csv(saving_path, index=False)

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
