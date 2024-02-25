import tqdm
import time
import psutil
import pandas as pd
import random

from river import metrics,stream
from EvOAutoML import classification
from ASML import WindowClassificationPerformanceEvaluator

import json

import argparse

import warnings
warnings.filterwarnings("ignore")

def main(dataset_name,run_count):

    # Currently, we are using default seed, but you can use a random seed for multiple runs.
    seed = 42 #random.randint(40,52)
    
    print(f"Loading dataset: {dataset_name}, Run Count:{run_count}, Random Seed:{seed}")


    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")

    x = df.drop('class', axis=1) 
    y = df['class']

    dataset = stream.iter_pandas(x, y)
    
    model = classification.EvolutionaryBaggingClassifier(population_size=10,
                                                     sampling_size=1,
                                                     sampling_rate=1000,
                                                     metric=metrics.Accuracy,
                                                     seed=seed)

    metric = metrics.Accuracy()
    
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

    scores_evo = []
    times_evo = []
    memories_evo = []

    for x, y in tqdm.tqdm(dataset):
        mem_before = psutil.Process().memory_info().rss
        start = time.time()

        y_pred = model.predict_one(x)  # make a prediction
        metric.update(y, y_pred)  # update the metric
        wcpe.update(y, y_pred) #windows Update
        model = model.learn_one(x,y)  # make the model learn
        
        end = time.time()
        mem_after = psutil.Process().memory_info().rss
        scores_evo.append(metric.get())
        iteration_mem = mem_after - mem_before
        memories_evo.append(iteration_mem)
        iteration_time = end - start
        times_evo.append(iteration_time)
    
    print(f"Accuracy on run {run_count} in {dataset_name}: {metric.get()}")
    
    save_record = {
        "model": "EvoAutoML",
        "dataset": dataset_name,
        "prequential_scores": scores_evo,
        "windows_scores": wcpe.get(),
        "time": times_evo,
        "memory": memories_evo
    }
    
    
    #file_name = f"{save_record['model']}_{save_record['dataset']}.json"
    file_name = f"{save_record['model']}_{save_record['dataset']}_run_{run_count}.json"
    
    #print("Result Saved path:",file_name)
    
    # To store the dictionary in a JSON file
    with open(f"saved_results_json/{file_name}", 'w') as json_file: # change temp to  saved_results_json for final run
        json.dump(save_record, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoAutoMl Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")
    parser.add_argument("--run_count", type=int, help="Number of the model run")
    args = parser.parse_args()
    main(args.dataset_name,args.run_count)