Given your folder structure and the provided Python script, I'll create a README file that includes instructions on how to run the script, along with a brief description of what it does. This README assumes that users have a basic understanding of Python and the command line.

---

# README for Automated Machine Learning Script

## Overview

This script facilitates the execution of various machine learning models on specified datasets. It supports a range of models including Hoeffding Adaptive Tree Classifier (HATC), Streaming Random Patches ensemble classifier (SRPC), Adaptive Random Forest classifier (ARFC), Online AutoML (OAML), AutoClass (AC), AutoStreamML (ASML), and EvoAutoML (EAML). The script is designed to run these models against datasets, allowing for multiple runs and providing flexibility in model and dataset selection.

## Folder Structure

Your project folder should include the following directories and files:

- `ASML/`, `AutoClass/`, `OAML/`: Directories containing scripts and resources specific to each model.
- `stream_datasets/`: Directory where your datasets are stored.
- `ac_run.py`, `asml_run.py`, `eaml_run.py`, `baseline_run.py`: Scripts for running specific models.
- `run_script.py`: The main script to run models on datasets.

## Prerequisites

- Python 3.6 or higher
- Required Python packages: `pandas`, `gama`, `river`, and others as needed by specific models.

## Installation

1. Ensure Python 3.6+ is installed on your system.
2. Install required Python packages using pip:

```bash
pip install pandas gama river
```

## Running the Script

To run the script, use the following command format in your terminal or command prompt:

```bash
python run_script.py --model_name MODEL_NAME --dataset_name DATASET_NAME --run_count RUN_COUNT
```

- `MODEL_NAME`: Short name of the model to run. Options: `asml`, `ac`, `oaml`, `eaml`, `hatc`, `srpc`, `arfc`.
- `DATASET_NAME`: Name of the dataset file (without the `.csv` extension) located in the `stream_datasets` folder.
- `RUN_COUNT`: Number of times to run the script for the given model and dataset.

### Example Command

```bash
python run_script.py --model_name asml --dataset_name electricity --run_count 1
```

This command runs the AutoStreamML (ASML) model on the `electricity.csv` dataset located in the `stream_datasets` folder, executing the script once.

## Note

- Ensure that the dataset files are correctly placed in the `stream_datasets` directory.
- Modify the script paths in the command construction within `run_script.py` if your folder structure differs from the expected setup.

## Environment Setup

Different models require different Python packages. Below are the requirements for each model:

### ASML and AutoClass Requirements

- numpy==1.24.4
- river==0.10.1
- For AutoClass, additionally: scipy==1.12.0

### OAML Requirements

- category_encoders==2.6.1
- liac_arff==2.5.0
- river==0.8.0
- scikit_learn==0.24.0
- scikit_multiflow==0.5.3
- stopit==1.1.2
- pandas==1.3.5
- numpy==1.20.1

### EAML Requirements

- EvoAutoML==0.0.14

To install these packages, use `pip` with the appropriate version numbers. For example, to set up the environment for ASML, you would run:

```bash
pip install numpy==1.24.4 river==0.10.1
```

Repeat this process for each model you plan to use, ensuring you meet all listed requirements.

## Jupyter Notebooks for Analysis

Several Jupyter notebooks are included for plotting results and making tables:

- `AutoRank_CD_Graph.ipynb`: Use this notebook to generate critical difference graphs for comparing model performances.
- `Final_Performence_Records_New.ipynb` and `Final_Performence_Records_Same_Config.ipynb`: These notebooks are used for analyzing and visualizing the final performance records of the models.
- `Plot_Result.ipynb`: Use this notebook for general plotting of results.

To use these notebooks, ensure you have Jupyter installed (`pip install jupyter`) and launch Jupyter Notebook or JupyterLab in the directory containing the notebooks. Each notebook contains specific instructions for use.

## Note

Ensure that the dataset files are correctly placed in the `stream_datasets` directory and that you have installed all required dependencies for the models you intend to use. Modify the script paths in the command construction within `run_script.py` if your folder structure differs from the expected setup.

--- 

This README provides a comprehensive guide to setting up your environment, running the script, and analyzing the results with Jupyter notebooks. Adjust the instructions as necessary to fit the specifics of your project or environment.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/6071087/947786e0-5188-467a-a6f8-cf917266e478/paste.txt
