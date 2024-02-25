import argparse
import concurrent.futures
import subprocess

# python run_script.py --model_name asml --dataset_name electricity --run_count 1

# HATC: HoeffdingA daptive Tree Classifier
# SRPC: Streaming Random Patches ensemble classifier
# ARFC: Adaptive Random Forest classifier
# OAML: Online AutoML
# AC:   AutoClass
# ASML: AutoStreamML
# EAML: EvoAutoML

# Define the function to run a script with a dataset name and optional model name and run count
def run_script(model_name, dataset_name, run_count=1):
    """
    Function to execute a given script for a specified dataset name, model name, and run count.
    """
    # Determine which script to run and construct the command accordingly
    if model_name == 'hatc' or model_name == 'srpc' or model_name == 'arfc':
        command = [
            'python', 'baseline_run.py', dataset_name, '--run_count', str(run_count), '--model_name', model_name
        ]
    elif model_name == 'oaml':
        command = [
            'python', 'OAML/oaml_ensemble.py', dataset_name, '5000', '5000', 'acc', 'acc', '60', 'evol', str(run_count)
        ]
    elif model_name == 'ac' or model_name == 'asml' or model_name == 'eaml':
        command = ['python', f'{model_name}_run.py', dataset_name]
    else:
        raise ValueError("Invalid script name provided.")

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print or return the output (stdout or stderr)
    if result.returncode == 0:
        print(f"Dataset {dataset_name}, Script {model_name}, Run {run_count}: Success\n{result.stdout}")
    else:
        print(f"Dataset {dataset_name}, Script {model_name}, Run {run_count}: Error\n{result.stderr}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run different scripts on datasets.")
parser.add_argument('--model_name', type=str, help='Short name of the script to run (asml,ac,oaml,eaml,hatc,srpc and arfc)')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset to run the script on (electricity,adult etc from stream_datasets folder)')
parser.add_argument('--run_count', type=int, default=1, help='Number of times to run the script (default is 1)')
args = parser.parse_args()

# Run the script once or multiple times based on the run_count argument
if args.run_count > 1:
    # Prepare a list of tasks for multiple runs
    tasks = [(args.model_name, args.dataset_name, run_count) for run_count in range(1, args.run_count + 1)]
    # Using ThreadPoolExecutor to run scripts concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda p: run_script(*p), tasks)
else:
    # Run the script once
    run_script(args.model_name, args.dataset_name, args.run_count)