import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from threading import Lock
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from datetime import datetime


# --------------------
def extract_binary_answer(response_text):
    """
    Extracts the value of the 'DNA' field (0 or 1) from the model's JSON response.
    Returns np.nan if parsing fails or the value is not 0 or 1.
    """
    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")

        json_str = match.group(0)
        result = json.loads(json_str)

        val = result.get("DNA", None)
        if str(val) in ["0", "1"]:
            return int(val)
        else:
            raise ValueError(f"Invalid value for DNA: {val}")

    except Exception as e:
        logger.error(f"Failed to parse response: {e}\nResponse was:\n{response_text}")
        return np.nan


# --------------------
# Unified API call with one prompt per note
def call_api(description, model, max_tokens=2000, temperature=0.2):
    """
    Sends the description to a local LLM via LM Studio API and returns 0 (nDNA), 1 (mtDNA), or np.nan.
    """
    url = f"{base_url}/v1/chat/completions"
    messages = [
        {
            "role": "system",
            "content": (
                "You are assisting with the classification of genetic mutations in patients with primary mitochondrial disorders. "
                "Based on the clinical presentation, determine whether the mutation is more likely to be in mitochondrial DNA (mtDNA) or nuclear DNA (nDNA). "
                "You should make a best guess based on the information provided"
                "There may be overlaps and uncertainties, as in real clinical cases."
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
        response = session.post(
            url,
            json={
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        logger.debug(f"Raw response text: {response.text}")

        if response.status_code == 200:
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            # print the content
            logger.debug(f"API response content: {content}")

            return extract_binary_answer(content)
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return np.nan
    except Exception as e:
        logger.error(f"API exception: {e}")
        return np.nan


# --------------------
# Append results to CSV safely
def append_result(result_row):
    with save_lock:
        pd.DataFrame([result_row]).to_csv(
            saving_path, mode="a", header=False, index=False
        )


# --------------------
# Analyze a single row
def analyze_row(row, model, max_tokens, temperature):
    global processed_counter
    description = row["description"]
    result_row = {"ID": row["ID"]}
    # logger.info(f"Processing ID {row['ID']} with description: {description}")

    mutation_result = call_api(description, model, max_tokens, temperature)
    result_row.update({"mutation": mutation_result})

    logger.info(f"Processed ID {row['ID']}: {result_row}")
    append_result(result_row)

    with counter_lock:
        processed_counter += 1
        percent = (processed_counter / total_patients) * 100
        logger.info(
            f"Progress: {processed_counter}/{total_patients} patients processed ({percent:.2f}%)"
        )




# Setup logging for info and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock to safely write to the output file from multiple threads
save_lock = Lock()

# Create a persistent HTTP session with retries to handle transient errors
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))

# Base URL of your local LM Studio server
base_url = "http://172.18.48.1:1234"

# --------------------
# CHANGE MODEL HERE to switch the LM used for inference

# model = "phi-4-mini-reasoning"  # Example: "deepseek-r1-distill-qwen-7b"
model = "deepseek-r1-distill-qwen-7b"
# model = "sauerkrautlm-gemma-2-9b-it-i1"  # fast

# model = "qwen2.5-7b-instruct"
# model = "wasamikirua-samantha2.0-qwen2.5-14b-ita" #maybe too big
# --------------------
# Parameters for the API call

max_tokens = 2000
temperature = 0.2

# Output file path depends on model name to keep separate results
# saving_path = f"/home/saram/PhD/Proximity_AI/database/labeled/amylo_casi_controlli/llm_annotations_{model}.csv"
saving_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/survey_answer_{model}.csv"
)

logger.info(f"Using model: {model}")
logger.info(f"Saving results to: {saving_path}")

# --------------------
# Load input data
file_path = "/home/saram/PhD/genomit_PMD/saved_results/survey/Description4Survey.csv"
data = pd.read_csv(file_path)


total_patients = len(data)
logger.info(f"Total patients to process: {total_patients}")

# Subset for testing
# data = data.iloc[:2]

# Filter out already processed entries
if os.path.exists(saving_path) and os.path.getsize(saving_path) > 0:
    logger.info("Loading previously processed entries...")
    processed_ids = pd.read_csv(saving_path)["ID"]
    data = data[~data["ID"].isin(processed_ids)]
    already_processed = len(processed_ids)
else:
    processed_ids = set()
    already_processed = 0
    logger.info("No previously processed entries found.")

processed_counter = already_processed
counter_lock = Lock()
# Print the number of already processed entries
logger.info(f"Already processed entries: {already_processed}")
# Print the number of entries to process
logger.info(f"Entries to process: {len(data)}")


# Create output file if missing
if not os.path.exists(saving_path):
    pd.DataFrame(columns=["ID", "mutation"]).to_csv(saving_path, index=False)

# --------------------

start_time = time.time()
# --------------------
# Run multithreaded processing
with ThreadPoolExecutor(max_workers=5) as executor:
    logger.info(f"Processing {len(data)} notes...")
    futures = [
        executor.submit(analyze_row, row, model, max_tokens, temperature)
        for _, row in data.iterrows()
    ]

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Thread error: {e}")

logger.info(f"Processing complete. Results saved to {saving_path}")


# --------------------
# Log Parameters Used
# --------------------


# Path for saving parameter log
params_log_path = (
    f"/home/saram/PhD/genomit_PMD/saved_results/survey/params_log_{model}.txt"
)

# Define what was used for logging
elapsed_time = time.time() - start_time if "start_time" in locals() else np.nan
sys_content = (
    "You are a medical assistant. Determine whether the mutation type is mitochondrial DNA (mtDNA) or nuclear DNA (nDNA) based on the provided information.",
)
prompt_example = f"""Here is the patient's clinical history:
        [ANAMNESIS TEXT] [omitted for brevity]\n\n

       Please respond with a JSON object in the following format:
    {{
    "DNA": 1
    }}

    Instructions:
    - Set "DNA" to 1 if the mutation is in mitochondrial DNA (mtDNA)
    - Set "DNA" to 0 if the mutation is in nuclear DNA (nDNA)

    Return only the JSON object. Do not include any explanation, comments, Markdown formatting, or additional text.
    """
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
