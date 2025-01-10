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

import os
import sys
import subprocess
from setuptools import setup, find_packages


def install_requirements():
    """Install the necessary dependencies for the project."""
    try:
        print("Installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during dependency installation: {e}")
        sys.exit(1)


def check_environment():
    """Check if the environment is set up correctly."""
    try:
        print("Checking environment...")
        if (
            "VIRTUAL_ENV" not in os.environ
            or not os.path.basename(os.environ["VIRTUAL_ENV"]) == ".env"
        ):
            print(
                "Warning: .env virtual environment is not active. It's recommended to activate it before proceeding."
            )
        else:
            print(f".env virtual environment is active: {os.environ['VIRTUAL_ENV']}")

        print("Environment is correctly set up.")
    except Exception as e:
        print(f"An error occurred during environment check: {e}")
        sys.exit(1)


def configure_dataset():
    """Prompt the user to configure the dataset."""
    try:
        print("Configuring dataset...")
        dataset_path = input(
            "Please provide the path to the PMD dataset (default: 'saved_results/df/df_Global.xlsx'): "
        )
        if dataset_path and os.path.exists(dataset_path):
            print(f"Using dataset from: {dataset_path}")
        else:
            print("Using default dataset from: 'saved_results/df/df_Global.xlsx'")
    except Exception as e:
        print(f"An error occurred during dataset configuration: {e}")
        sys.exit(1)


def main():
    """Main function to perform environment setup."""
    try:
        check_environment()
        install_requirements()
        configure_dataset()
        print("Setup completed successfully.")
    except Exception as e:
        print(f"An error occurred in the setup process: {e}")
        sys.exit(1)


# Set up project metadata and configuration
setup(
    name="genomit_PMD",
    version="1.0.0",
    description=(
        "A Python framework for machine learning-based classification of mutation types "
        "in Primary Mitochondrial Diseases (PMDs) using non-invasive data."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sara Mazzucato",
    author_email="sara.mazzucato.phd@gmail.com",
    url="https://github.com/saramazz/genomit_PMD.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "initialize_environment=ml_pipeline.scripts.initialize:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://github.com/saramazz/genomit_PMD",
        "Source": "https://github.com/saramazz/genomit_PMD",
        "Bug Tracker": "https://github.com/saramazz/genomit_PMD/issues",
    },
)

if __name__ == "__main__":
    main()
