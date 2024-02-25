# AutoStreamML (ASML) Code

## Overview

This code facilitates the execution of ASML and baseline models on datasets. It includes Online AutoML (OAML), AutoClass (AC), AutoStreamML (ASML), EvoAutoML (EAML), and state-of-the-art online learning algorithms such as Hoeffding Adaptive Tree Classifier (HATC), Streaming Random Patches ensemble classifier (SRPC), and Adaptive Random Forest classifier (ARFC). The models are designed to run online learning against given datasets, allowing for multiple runs and providing flexibility in model and dataset selection.

## Folder Structure

Our project folder should include the following directories and files:

- `ASML/`: Our proposed method.
- `AutoClass/`, `OAML/`: Directories containing scripts and resources specific to each model.
- `run_script.py`: The main script to run models on datasets.
- `stream_datasets/`: Directory where datasets are stored.
- `saved_results_json/` and `multi_run_server_json`: Directories where running raw results are saved.

Note: In the folder `multi_run_server_json`, raw results are not present due to space issues. If included, it would become around a 6GB file, and GitHub would fail to upload that.

## Prerequisites

- Python 3.8 or higher (We ran all our experiments in Python 3.8.10).
- Required Python packages:
    - For every model, `requirements.txt` files are given in their respective folders; you just need to run them one by one.
    - All the models support `river==0.10.1`, except the OAML model needs `river==0.8.0`.
    - Before running the code, you must install some helper libraries such as `tqdm`, `psutil`, etc. These come preinstalled with Anaconda.

## Installation

1. Ensure Python 3.8+ is installed on your system.
2. Install the required Python packages using pip and the `requirements.txt` files:

```bash
pip install -r ASML/requirements.txt
pip install -r AutoClass/requirements.txt
pip install -r OAML/requirements.txt
pip install tqdm psutil
pip install EvoAutoML==0.0.14
```

## Running the Script

To run the script, use the following command format in your terminal or command prompt:

```bash
python run_script.py --model_name MODEL_NAME --dataset_name DATASET_NAME --run_count RUN_COUNT
```

- `MODEL_NAME`: The short name of the model to run. Options include `asml`, `ac`, `oaml`, `eaml`, `hatc`, `srpc`, `arfc`.
- `DATASET_NAME`: The name of the dataset file (without the `.csv` extension) located in the `stream_datasets` folder.
- `RUN_COUNT`: The number of times to run the script for the given model and dataset. (Note: If you want to run multiple times, then you need to change the default seed to a random seed in each script.)

### Example Command

```bash
python run_script.py --model_name asml --dataset_name electricity --run_count 1
```

This command runs the AutoStreamML (ASML) model on the `electricity.csv` dataset located in the `stream_datasets` folder, executing the script once.

## Jupyter Notebooks for Analysis

Several Jupyter notebooks are included for plotting results and making tables:

- `AutoRank_CD_Graph.ipynb`: Use this notebook to generate critical difference graphs for comparing model performances.
- `Final_Performance_Records_New.ipynb` and `Final_Performance_Records_Same_Config.ipynb`: These notebooks are used for analyzing and visualizing the final performance records of the models.
- `Plot_Result.ipynb`: Use this notebook for general plotting of results.

To use these notebooks, ensure you have Jupyter installed (`pip install jupyter`) and launch Jupyter Notebook or JupyterLab in the directory containing the notebooks. Each notebook contains specific instructions for use.

## Note

- Ensure that the dataset files are correctly placed in the `stream_datasets` directory.
- Modify the script paths in the command construction within `run_script.py` if your folder structure differs from the expected setup.
- Currently, all the run files are stored in the `temp` folder; you need to change it to where you want to store the run results.