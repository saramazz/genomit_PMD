# Configuration file with global variables
import os

# Set the global path for the project
# Get the directory of the current Python script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory of the script as the global path
global_path = os.path.dirname(script_directory)
print("Global path:", global_path)

# Define the path for saved results folder
saved_result_path = os.path.join(global_path, "saved_results")  # Directory for saved results
# print('Save result path:', saved_result_path)

saved_result_path_classification = os.path.join(saved_result_path, "classifiers_results")
# Optional alternative path (uncomment and modify if needed)
# saved_result_path_classification = (
#     os.path.join(saved_result_path, "classifiers_results/experiments") or best_model_idx
# )

saved_result_path_survey = os.path.join(saved_result_path, "survey")

# Print the classification results path for debugging purposes
print("Save result path classification:", saved_result_path_classification)

'''
# Import cProfile for performance profiling
import cProfile

# Define the path to the main script for classification tasks
script_path = os.path.join(global_path, "src", "main_classification_DNA.py")

# Uncomment the following line to profile the script execution
cProfile.run("exec(open(script_path).read())")
'''