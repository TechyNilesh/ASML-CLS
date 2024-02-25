import psutil
import time
import tqdm
from river import stream, metrics, linear_model,naive_bayes,tree,neighbors,preprocessing
from ASML import AutoStreamClassifier
from ASML import WindowClassificationPerformanceEvaluator
import pandas as pd
import json
import argparse
import random

import warnings

warnings.filterwarnings("ignore")


def main(dataset_name, run_count, EW=1000, ES=3, B=10):
    
    # Currently, we are using default seed, but you can use a random seed for multiple runs.
    seed = 42 #random.randint(42,52)

    print(
        f"Loading dataset: {dataset_name}, Run Count:{run_count}, Random Seed:{seed}")
    print(f"Current Hyperparameters: EW - {EW}, ES - {ES}, B - {B}")

    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")

    x = df.drop("class", axis=1)
    y = df["class"]

    dataset = stream.iter_pandas(x, y)

    ASC = AutoStreamClassifier(config_dict=None, #config_dict
        exploration_window=EW, # Window Size
        prediction_mode="ensemble", #change 'best' if you want best model prediction 
        budget=B,# How many pipelines run concurrently
        ensemble_size=ES, # Ensemble size 
        metric=metrics.Accuracy(), # Online metrics
        verbose=False,
        seed=seed, # Random/Fixed seed
    )

    online_metric = metrics.Accuracy()
    
    
    # WCPE for plotting the results in line graph
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

    scores = []
    times = []
    memories = []
    for x, y in tqdm.tqdm(dataset, leave=True):
        mem_before = psutil.Process().memory_info().rss # Recording Memory
        start = time.time()  # Recording Time
        y_pred = ASC.predict_one(x)  # Predict/Test
        s = online_metric.update(y, y_pred).get() # Update Metrics
        # windows Update
        wcpe.update(y, y_pred)
        ASC.learn_one(x, y) # Online Learning
        end = time.time()
        mem_after = psutil.Process().memory_info().rss
        scores.append(s)
        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        iteration_time = end - start
        times.append(iteration_time)

    print(
        f"Accuracy on run {run_count} in {dataset_name}: {online_metric.get() * 100:0.2f}%"
    )
    
    # saving results in dict
    save_record = {
        "model": "AutoStreamML",
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": wcpe.get(),
        "time": times,
        "memory": memories
    }

    #file_name = f"{save_record['model']}_{save_record['dataset']}.json"
    file_name = f"{save_record['model']}_{save_record['dataset']}_run_{run_count}.json"    
    # To store the dictionary in a JSON file
    
    # To store the dictionary in a JSON file
    with open(f"temp/{file_name}", 'w') as json_file:  # change temp to  saved_results_json for final run
        json.dump(save_record, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoStreamML Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")
    parser.add_argument("--run_count", type=int, default=0, help="Number of the model run")
    parser.add_argument("--exploration_window", type=int, default=1000, help="Exploration Window")
    parser.add_argument("--ensemble_size", type=int, default=3, help="Ensemble Size")
    parser.add_argument("--budget", type=int, default=10, help="Budget")
    args = parser.parse_args()
    main(
        args.dataset_name,
        args.run_count,
        args.exploration_window,
        args.ensemble_size,
        args.budget,
    )
