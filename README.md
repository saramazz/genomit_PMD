# ML-Based Mutation Classification in Primary Mitochondrial Diseases

This repository provides a Python-based framework for applying machine learning (ML) techniques to classify mutation types (mitochondrial DNA vs. nuclear DNA) in patients with Primary Mitochondrial Diseases (PMDs). By leveraging basic phenotype data, our approach seeks to improve diagnostic accuracy without relying on invasive histological or genetic testing.

## Overview  

Primary Mitochondrial Diseases (PMDs) are a diverse group of rare inherited metabolic disorders caused by mutations in either mitochondrial DNA (mtDNA) or nuclear DNA (nDNA). Currently, diagnosing PMDs involves complex interpretation of clinical, biochemical, and histological data, which limits accessibility to accurate diagnoses.

This project investigates the use of machine learning to automate mutation type classification based solely on phenotypic data. The framework evaluates four ML classifiers—XGBoost, Random Forest, Decision Tree, and Support Vector Machine—on a multicenter dataset comprising 1046 patients from nine Italian centers.

### Key Features  




## Setup  

### Installation  

To install the required dependencies, run:  

```bash  
pip install -r requirements.txt  
```  

### Configuration  

Ensure your environment is set up correctly before running the models:  

```bash  
python setup.py  
```  

This script initializes your environment, prompts for dataset configuration, and ensures all dependencies are correctly installed.  

## Quick Start  

### Running the Model  

To train and evaluate the models, follow these steps:  

```python  
from ml_pipeline import load_data, train_model, evaluate_model  

# 1. Load the dataset  
data = load_data('data/PMD_dataset.csv')  

# 2. Train the Random Forest model  
model = train_model(data, model_type='random_forest', balance=True)  

# 3. Evaluate the model on the test set  
results = evaluate_model(model, test_data=data['test'])  
print(results)  
```  

### Searching for Optimal Parameters  

To search for the best hyperparameters using grid search:  

```python  
from ml_pipeline import hyperparameter_search  

best_params = hyperparameter_search('random_forest', data['train'])  
print("Optimal Parameters:", best_params)  
```  

### Inspecting Model Performance  

Visualize model performance:  

```python  
from ml_pipeline import plot_results  

plot_results(results)  
```  

## Dataset  

The dataset used in this study consists of 1046 PMD patients. It is divided into training and test sets, with features derived from clinical phenotype data.  

For more information on data preprocessing and feature selection, refer to the [documentation](#).  

## Contributing  

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.  

<!---  ## License 

<> This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  --->  

## Acknowledgments  

We thank the nine Italian centers for providing the dataset and the expert physicians in mitochondrial medicine for their valuable insights.  

## Contact  

For questions or further information, please contact:  

**Sara Mazzucato**  
PhD Candidate  
Email: sara.mazzucato.phd@gmail.com
```
