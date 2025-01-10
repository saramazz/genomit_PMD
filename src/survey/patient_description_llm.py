import os
import sys
import re
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import openai
from openai import OpenAI

# Add the project directory to the system path

from dotenv import load_dotenv

load_dotenv()  # Carica le variabili d'ambiente da un file .env

# Ora recupera l'API key dalla variabile d'ambiente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config_api import API_KEY

# API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# check if the api was imported correctly
if not API_KEY:
    raise ValueError("API key not found, configure OPENAI_API_KEY.")
print(API_KEY)

# Define paths
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(script_directory)
global_path = os.path.dirname(parent_path)

saved_result_path = os.path.join(global_path, "saved_results")
saved_result_path_survey = os.path.join(saved_result_path, "survey")


"""
LOAD DATA
"""

df_path = os.path.join(
    saved_result_path_survey, "df_test_human_friendly_best.csv"
)  # description of the patients in a human friendly way
df = pd.read_csv(df_path)

print(df.head())


"""
APPROACH WITHOUT LLM
"""
# Ask if the user whant to proceed with the approach without LLM
proceed = input("Do you want to proceed with the approach without LLM? (y/n): ")

if proceed == "n":
    print("Exiting the script...")
    sys.exit(0)
# Prepare the results list
results = []


# Function to process and save non-NaN values for a specific patient
def process_patient_data(patient_number, patient_data):
    patient_id = patient_data["id paziente"]
    description = []
    for column, value in patient_data.items():

        if (
            pd.notna(value)
            and column != "id paziente"
            and value
            not in {"998.0", "998", 998, "997.0", "997", 997, "9999", "9999.0", 9999}
        ):

            # Append each field as "Key: Value" in a readable format
            description.append(f"{column}: {value}")
    # Create a human-friendly description with line breaks
    description_text = "\n".join(description)
    # substitute in description the .0 with empty string
    description_text = re.sub(r"\.0", "", description_text)
    # add a column of the patient_true_id to the results
    # patient_true_id = patient_data["patient_id"]
    # Append the sequential patient ID and description to the results
    results.append({"patient_id": f"{patient_id}", "description": description_text})
    # substitute in description the .0 with empty string
    description_text = re.sub(r"\.0", "", description_text)


# Iterate through all patients and process their data
for index, row in df.iterrows():
    process_patient_data(patient_number=index, patient_data=row)

# Save the results to a CSV file
# output_path = os.path.join(saved_result_path_survey, f"processed_patient_data_human_friendly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
output_path = os.path.join(
    saved_result_path_survey, f"description_patient_data_test_set.csv"
)

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
print(results_df)
print("Dataset dimension: ", df.shape)


"""
APPROACH WITH LLM
"""
import pandas as pd
import os
from datetime import datetime


# Define the GPT prompt template
def create_prompt(row):
    prompt = (
        "You are a helpful assistant tasked with creating patient descriptions based on the provided data. "
        "Do not consider NaN, 998, 997, or 9999 values, and respond in Italian, do not include the id of the patient. Do not add other text to the answer "
        "Use integers for numbers (e.g., use 4 instead of 4.0)."
        "\n\n"
        "Data to be elaborated:\n"
        f"{row}\n\n"
        "Example of data for a patient:\n"
        f"{df.iloc[0]}\n\n"
        "Example of correct description format:\n"
        f"{generate_patient_description(df.iloc[0])}\n"
    )
    return prompt


# Function to generate a description of non-NaN values for a specific patient
def generate_patient_description(patient_data):
    return "\n".join(
        [
            f"{column}: {value}"
            for column, value in patient_data.items()
            if pd.notna(value)
        ]
    )


# Function to generate descriptions with the LLM
def create_descriptions(index, row, id, max_tokens=300, temperature=0.5):
    description = generate_patient_description(row)
    prompt = create_prompt(row.to_dict())

    messages = [{"role": "user", "content": prompt}]
    description_llm = ""

    # Call the LLM API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Extract the content from the LLM response
    description_llm = response.choices[0].message.content.strip()

    return {"patient_id": f"Patient_{id}", "description": description_llm}


# Apply the function to the dataset
def process_dataset_and_save(df, output_path):
    results = []

    for index, row in df.iterrows():
        # print the patient id
        print(f"Processing patient {index + 1}...")
        # extract the id of the patient
        id = row["id paziente"]
        result = create_descriptions(index, row, id)
        results.append(result)

    # Save the results to a CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    # print the results
    print(results_df)

    print(f"Descriptions saved to {output_path}")
    return result


output_path = os.path.join(
    saved_result_path_survey, f"description_patient_data_test_set_llm.csv"
)


proceed = input("Do you want to proceed with the approach LLM? (y/n): ")

if proceed == "n":
    print("Exiting the script...")
    sys.exit(0)

# Process the dataset and save descriptions
result = process_dataset_and_save(df, output_path)

# save in a log file the details of the number of patients, model used, max tokens and the time of the run
log_file_path = os.path.join(
    saved_result_path_survey, "log_file_description_patient_data_test_set_llm.txt"
)
with open(log_file_path, "w") as log_file:
    log_file.write(f"Number of patients: {len(df)}\n")
    log_file.write("Model used: gpt-4o\n")
    log_file.write("Max tokens: 300\n")
    log_file.write("Temperature: 0.5\n")
    log_file.write(f"Time: {datetime.now()}\n")

    log_file.write(f"Prompt: {result}\n")
