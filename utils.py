import json
import torch

def load_config(config_file="config.json"):
    """
    Load parameters from a JSON configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


class EarlyStopping:
    """
    Implements early stopping to prevent overfitting.
    Monitors validation loss and stops training if it doesn't improve after a certain number of epochs.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        """
        Step function to evaluate if the model's performance on the validation set has improved.

        Args:
            val_loss (float): The current validation loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
