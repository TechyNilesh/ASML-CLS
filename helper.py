from collections import deque
import matplotlib.pyplot as plt
import pandas as pd


def range_gen(min_n,max_n,step=1,float_n=False):
    if float_n:
        return [min_n + i * step for i in range(int((max_n - min_n) / step) + 1)]
    return list(range(min_n,max_n+1,step))

class RollingWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)

    def add_item(self, item):
        if len(self.window) >= self.window_size:
            self.window.popleft()
        self.window.append(item)

    def get_items(self):
        return list(self.window)
    def is_full(self):
        return len(self.window) == self.window_size
    def is_empty(self):
        return len(self.window) == 0

def plot_smoothed_graphs(scores, memories, times, window_size, metric_name='Metric'):
    def smooth_data(data):
        smoothed_data = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = i + 1
            window_data = data[start:end]
            smoothed_data_point = sum(window_data) / len(window_data)
            smoothed_data.append(smoothed_data_point)
        return smoothed_data

    # Smooth the data
    smoothed_scores = smooth_data(scores)
    smoothed_memories = smooth_data(memories)
    smoothed_times = smooth_data(times)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot smoothed metric scores
    axes[0].plot(range(1, len(smoothed_scores) + 1), smoothed_scores, color='green')
    axes[0].set_xlabel('Data Point')
    axes[0].set_ylabel(f'{metric_name}')
    axes[0].set_title(f'{metric_name} Scores for Each Data Point')
    axes[0].grid(True)

    # Plot smoothed memory usage rates
    axes[1].plot(range(1, len(smoothed_memories) + 1), smoothed_memories, color='blue')
    axes[1].set_xlabel('Data Point')
    axes[1].set_ylabel('Memory Usage Rate')
    axes[1].set_title('Memory Usage Rate for Each Data Point')
    axes[1].grid(True)

    # Plot smoothed time usage rates
    axes[2].plot(range(1, len(smoothed_times) + 1), smoothed_times, color='red')
    axes[2].set_xlabel('Data Point')
    axes[2].set_ylabel('Time Usage Rate (Sec)')
    axes[2].set_title('Time Usage Rate for Each Data Point')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_columns_distribution(df):
    """
    Plot the distribution of all columns in a DataFrame using Kernel Density Estimate (KDE).

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.

    Returns:
    None
    """
    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Plot the KDE distribution for each column
    for column in df.columns:
        df[column].plot(kind='kde', title=f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.show()

def plot_smoothed_scores(score_lists, window_size, model_names):
    def smooth_data(data):
        smoothed_data = []
        for i in range(len(data)):
            start = max(0, i - window_size)
            end = i + 1
            window_data = data[start:end]
            smoothed_data_point = sum(window_data) / len(window_data)
            smoothed_data.append(smoothed_data_point)
        return smoothed_data

    num_models = len(score_lists)

    # Create a subplot for the smoothed metric scores
    plt.figure(figsize=(10, 6))
    
    for i in range(num_models):
        # Smooth the data for each model
        smoothed_scores = smooth_data(score_lists[i])

        # Plot smoothed metric scores
        plt.plot(range(1, len(smoothed_scores) + 1), smoothed_scores, label=model_names[i])

    plt.xlabel('Data Point')
    plt.ylabel('Metric Score')
    plt.title('Smoothed Metric Scores for Each Data Point')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class List(list):
    def last_index(self):
        return len(self)-1