### Standard Library:
import os
import pickle

def save_variable(variable, name, save_path):
    final_name = os.path.join(save_path, name + ".pickle")
    with open(final_name, "wb") as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)


# Function to load a pickle variable
def load_pickle_variable(name, load_path):
    file_path = os.path.join(load_path, name + ".pickle")
    with open(file_path, "rb") as file:
        variable = pickle.load(file)
    return variable


# Function to load and print the contents of a pickle file
def load_and_print_pkl(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"\nContents of the pickle file '{os.path.basename(file_path)}':")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"{key}: {value}")
            else:
                print(data)
    else:
        print(f"The file {file_path} does not exist.")


# Iterate through all files in the directory and print the contents of each pickle file
def print_all_pkl_files(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(directory_path, file_name)
            load_and_print_pkl(file_path)
