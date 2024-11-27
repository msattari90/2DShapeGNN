import json
import torch
import os

def load_config(config_file="config.json"):
    """
    Load parameters from a JSON configuration file.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

class EarlyStopping:
    """
    Implements early stopping to prevent overfitting by monitoring validation loss.
    """
    def __init__(self, patience, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        """
        Evaluate if the model's validation loss improved.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
