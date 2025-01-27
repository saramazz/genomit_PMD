# Copyright 2023 Sara Mazzucato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Configuration file with global variables
import os

# Set the global path for the project
# Get the directory of the current Python script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory of the script as the global path
global_path = os.path.dirname(script_directory)
print("Global path:", global_path)

important_vars_path = os.path.join(
    global_path, "variables_mapping/important_variables.xlsx"
)


# Define the path for saved results folder
saved_result_path = os.path.join(
    global_path, "saved_results"
)  # Directory for saved results
print("Save result path:", saved_result_path)

saved_result_path_classification = os.path.join(
    saved_result_path, "classifiers_results"
)

#saving folder for classification models
saved_result_path_classification_models = os.path.join(
    saved_result_path_classification, "models"
)

saved_result_path_survey = os.path.join(saved_result_path, "survey")

#if not existing, create the directories for the saved results
if not os.path.exists(saved_result_path) :
    os.makedirs(saved_result_path)
if not os.path.exists(saved_result_path_classification) :
    #also for classification and survey
    os.makedirs(saved_result_path_classification)
if not os.path.exists(saved_result_path_survey) :
    os.makedirs(saved_result_path_survey)
if not os.path.exists(saved_result_path_classification_models) :
    os.makedirs(saved_result_path_classification_models)




"""
# Import cProfile for performance profiling
import cProfile

# Define the path to the main script for classification tasks
script_path = os.path.join(global_path, "src", "main_classification_DNA.py")

# Uncomment the following line to profile the script execution
cProfile.run("exec(open(script_path).read())")
"""
