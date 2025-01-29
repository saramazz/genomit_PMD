

# Local Module Imports
from config import (
    global_path,
    saved_result_path,
    important_vars_path,
)
from utilities import *
from processing import *
from plotting import *

import datetime
import os
import sys

# Constants and Paths
GLOBAL_DF_PATH = os.path.join(saved_result_path, "df", "df_Global_preprocessed.csv")
SAVING_PATH = os.path.join(
    saved_result_path, "dataset_distribution"
)
BEST_PATH = os.path.join(saved_result_path_classification, "best_model")

df_classification_path= os.path.join(BEST_PATH, "df_classification.csv") 

# Create the directory if it does not exist
if not os.path.exists(SAVING_PATH):
    os.makedirs(SAVING_PATH)\

def read_features_from_file(file_path):
    # Initialize an empty list to store the features
    features = []
    
    # Use 'with' to open the file and ensure it gets closed properly
    with open(file_path, "r") as f:
        # Iterate over each line in the file
        for line in f:
            # Strip newline characters from the line and add to the list
            features.append(line.strip())
    
    return features

file_path = os.path.join(BEST_PATH, "features.txt")
features_list = read_features_from_file(file_path)

#add gendna to the features list
features_list.append('gendna')

# Print the list of features
print(features_list)


type_data= ['all', 'train_test','test']

'''
Print distribution of sex, age at onset (aao) and cage and number of patients for each hospital

print clinical diagnosis
'''

# Redirect the standard output to a file
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#file_name = f"dataset_distribution_{current_datetime}.txt"
file_name = f"dataset_distribution.txt"
# Redirect the standard output to the file
sys.stdout = open(os.path.join(SAVING_PATH, file_name), "w")



for type_db in type_data:
    # Load the global dataframe
    df_global = pd.read_csv(GLOBAL_DF_PATH)

    if 'train' in type_db:
        df_classification = pd.read_csv(df_classification_path)
        patient_classification=df_classification['subjid']
        df= df_global[df_global['subjid'].isin(patient_classification)]
        #maintain only the columns in features_list
        df = df[features_list]
    elif "test" in type_db:
        test_subjects_path = os.path.join(
            saved_result_path_classification, "saved_data", "test_subjects_final.pkl"
        )  # "test_subjects_num.pkl")#

        if not os.path.exists(test_subjects_path):
            raise FileNotFoundError(
                "Saved test subjects file is missing. Test subjects must be defined first."
            )

        with open(test_subjects_path, "rb") as f:
            test_subjects_ids = pickle.load(f)
        
        df= df_global[df_global['subjid'].isin(test_subjects_ids)]
        #maintain only the columns in features_list
        df = df[features_list]
    else:
        df=df_global
    
    # Display the dimensions and columns of the DataFrame
    nRow, nCol = df.shape
    print("Type of data: ", type_db)
    print("The DataFrame contains {0} rows and {1} columns.".format(nRow, nCol))

    plot_missing_values(
    df,  SAVING_PATH, f"Histogram_MissingValues_{type_db}.png")

    #remove the columns with 0 missing values
    no_missing_values_columns = df.columns[df.isnull().mean() == 0]
    #drop the columns with 0 missing values
    df_missing = df.drop(no_missing_values_columns, axis=1)
    plot_missing_values(
    df_missing,  SAVING_PATH, f"Histogram_MissingValues_no0_{type_db}.png")    
    
    # print data info

    try:

    
        df_aao = df["aao"].dropna()
    


        # Calculate percentages for aao below and above 16 years old
        aao_below_16 = df_aao[df_aao < 16]
        aao_below_16_percentage = (len(aao_below_16) / len(df_aao)) * 100
        aao_above_16 = df_aao[df_aao >= 16]
        aao_above_16_percentage = (len(aao_above_16) / len(df_aao)) * 100

        print(
            f"Percentage of subjects with Age at Onset (aao) below 16 years old: {aao_below_16_percentage:.2f}%"
        )
        print(
            f"Percentage of subjects with Age at Onset (aao) above 16 years old: {aao_above_16_percentage:.2f}%"
        )
    except:
        print("No Age at Onset column")

    try:

        df_cage = df["cage"].dropna()
            


        # Calculate percentages for cage below and above 16 years old
        cage_below_16 = df_cage[df_cage < 16]
        cage_below_16_percentage = (len(cage_below_16) / len(df_cage)) * 100
        cage_above_16 = df_cage[df_cage >= 16]
        cage_above_16_percentage = (len(cage_above_16) / len(df_cage)) * 100

        print(
            f"Percentage of subjects with Calculated Age (cage) below 16 years old: {cage_below_16_percentage:.2f}%"
        )
        print(
            f"Percentage of subjects with Calculated Age (cage) above 16 years old: {cage_above_16_percentage:.2f}%"
        )
    except:
        print("No Calculated Age column")
    
    try:

            # Filter out NaN values for each column
        df_sex = df["sex"].dropna()

        # Calculate sex distribution percentages
        sex_counts = df_sex.value_counts()
        total_count = len(df_sex)

        percentage_male = (sex_counts["m"] / total_count) * 100
        percentage_female = (sex_counts["f"] / total_count) * 100

        print(f"Percentage male: {percentage_male:.2f}%")
        print(f"Percentage female: {percentage_female:.2f}%")
        print(f"Total percentage: {percentage_male + percentage_female:.2f}%")
    except:
        print("No sex column")

    # sobstitute using this mapping in the column clindiag__decod
    clindiag_mapping = {
        "C01": "MELAS",
        "B01": "CPEO",
        "A02": "ADOA",
        "A01": "LHON",
        "C04": "Leigh syndrome",
        "C19": "Encephalopathy",
        "B02": "CPEO plus",
        "C03": "MERRF",
        "B03": "MiMy (without PEO)",
        "E": "unspecified mitochondrial disorder",
        "C06": "Kearns-Sayre-Syndrome (KSS)",
        "C05": "NARP",
        "C18": "Encephalomyopathy",
        "C02": "MIDD",
        "C17": "Other mitochondrial multisystem disorder",
        "C07": "SANDO/MIRAS/SCAE",
        "F": "asymptomatic mutation carrier",
        "D01": "Isolated mitochondrial Cardiomyopathy",
        "A03": "other MON",
        "C08": "MNGIE",
        "C16": "LBSL",
        "C": "Mitochondrial Multisystem Disorders",
        "C09": "Pearson syndrome",
        "C12": "Wolfram-Syndrome (DIDMOAD-Syndrome)",
        "D05": "Other mitochondrial mono-organ disorder",
    }

    try: 
        df_clindiag = df
        df_clindiag["clindiag__decod"] = df_clindiag["clindiag__decod"].map(clindiag_mapping)


        # Create a histogram using the text values

        plt.figure(figsize=(10, 6))

        df_clindiag["clindiag__decod"].value_counts().plot(kind="bar", color="skyblue")

        # plt.title("Histogram of clindiag__decod")

        plt.xlabel("Clinical Diagnosis")

        plt.ylabel("Number of patients")
        # increase font size
        plt.xticks(fontsize=12)

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVING_PATH, f"clindiag_hist_{type_db}.png"),
            format="png",
            bbox_inches="tight",
        )
        plt.close()
    except:
        print("No Clinical Diagnosis column")

    #print gendna distribution


    # Create 'nDNA' and 'mtDNA' classes
    df["nDNA"] = df["gendna"].apply(
        lambda x: "nDNA" if x in [4, 6, 8] else None
    )
    df["mtDNA"] = df["gendna"].apply(
        lambda x: "mtDNA" if x in [5, 7] else None
    )

    # Combine 'nDNA' and 'mtDNA' classes into 'gendna_type' column
    df["gendna_type"] = df.apply(
        lambda row: row["nDNA"] if row["nDNA"] is not None else row["mtDNA"], axis=1
    )

    
    #print distribution of patients for each hospital
    if 'Hospital' in df.columns:
        df_hospital = df["Hospital"].dropna()
        print(df_hospital.value_counts())
    else:   
        print("No Hospital column")



