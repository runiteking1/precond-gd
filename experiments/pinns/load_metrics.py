import pickle
import numpy as np

def load_metrics(filename):
    """
    Loads metrics history from a pickle file.

    Args:
        filename (str): The path to the .pkl file.

    Returns:
        dict: The loaded metrics_history dictionary, or None if loading fails.
    """
    with open(filename, 'rb') as f: # Open in binary read mode ('rb')
        loaded_metrics = pickle.load(f)
    print(f"Successfully loaded metrics from: {filename}")
    return loaded_metrics
