import tqdm
import time
import psutil
import json
import pandas as pd

from river import metrics,stream,preprocessing,linear_model,ensemble
from river import tree

import warnings
warnings.filterwarnings("ignore")

import argparse
import random
from ASML import WindowClassificationPerformanceEvaluator

# Extract the name of the model from the model class
def extract_model_short_form(model):
    input_string = type(model).__name__
    uppercase_letters = []
    for char in input_string:
        if char.isupper():
            uppercase_letters.append(char)
    return ''.join(uppercase_letters)

def main(dataset_name,run_count,model_name):

    seed = 42 #random.randint(42,52) # Currently, we are using default seed, but you can use a random seed for multiple runs.

    # Selecting a model from a set of baseline models
    if model_name=='HATC':
        model_raw = tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    elif model_name=='LBC':
        model_raw = ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingAdaptiveTreeClassifier(),seed=seed)
    elif model_name=='SRPC':
        model_raw = ensemble.SRPClassifier(n_models=3,seed=seed)
    elif model_name=='ARFC':
        model_raw = ensemble.AdaptiveRandomForestClassifier(n_models=3,seed=seed)

    print(f"Model Name: {extract_model_short_form(model_raw)}")
    
    print(f"Loading dataset: {dataset_name}, Run Count:{run_count}, Random Seed:{seed}")

    # We are using Standerd Scaler for all of the model preprocessing.
    model = preprocessing.StandardScaler() | model_raw

    # Reading Datasets
    df = pd.read_csv(f"stream_datasets/{dataset_name}.csv")
    
    x = df.drop('class', axis=1) 
    y = df['class']
    
    # converting dataframe to stream
    dataset = stream.iter_pandas(x, y)
    
    # storing the results    
    scores = []
    times = []
    memories = []
    metric = metrics.Accuracy()
    
    # WCPE for plotting the results in line graph
    wcpe = WindowClassificationPerformanceEvaluator(metric=metrics.Accuracy(),
                                                    window_width=1000,
                                                    print_every=1000)

    for x, y in tqdm.tqdm(dataset,leave=False):
        mem_before = psutil.Process().memory_info().rss # Recording Memory
        start = time.time() # Recording Time
        try:
            y_pred = model.predict_one(x) # Predict/Test
            s = metric.update(y,y_pred).get() # Update Metrics
            wcpe.update(y, y_pred) # windows Update
            model.learn_one(x, y) # Online Learning
        except:
            s=0
            continue
        end = time.time()
        mem_after = psutil.Process().memory_info().rss 
        scores.append(s)
        iteration_mem = mem_after - mem_before
        memories.append(abs(iteration_mem))
        iteration_time = end - start
        times.append(abs(iteration_time))

    # saving results in dict
    save_record = {
        "model": extract_model_short_form(model_raw),
        "dataset": dataset_name,
        "prequential_scores": scores,
        "windows_scores": wcpe.get(),
        "time": times,
        "memory": memories
    }
    
    print(f"{extract_model_short_form(model_raw)}: Accuracy on {dataset_name}: {metric.get()}")
    
    #file_name = f"{save_record['model']}_{save_record['dataset']}.json"
    file_name = f"{save_record['model']}_{save_record['dataset']}_run_{run_count}.json" # file name for save
    
    # To store the dictionary in a JSON file
    with open(f"temp/{file_name}", 'w') as json_file: # change temp to  saved_results_json for final run
    #with open(f"{file_name}", 'w') as json_file:
        json.dump(save_record, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Run Script")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset file (without extension)")
    parser.add_argument("--run_count", type=int, help="Number of the model run")
    parser.add_argument("--model_name", type=str, help="Name of the dataset file (without extension)")
    args = parser.parse_args()
    main(args.dataset_name,args.run_count,args.model_name)