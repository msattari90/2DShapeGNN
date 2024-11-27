import json
import torch
import os

def load_config(config_file="config.json"):
    """
    Load configuration parameters from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file (default: "config.json").
        
    Returns:
        dict: Configuration parameters loaded from the JSON file.
        
    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

class EarlyStopping:
    """
    Implements early stopping to monitor validation loss and prevent overfitting.
    
    Attributes:
        patience (int): Number of epochs to wait before stopping when no improvement.
        delta (float): Minimum change in validation loss to qualify as an improvement.
        best_loss (float): Best validation loss observed so far.
        counter (int): Number of consecutive epochs with no improvement.
        early_stop (bool): Flag indicating whether training should stop.
    """
    def __init__(self, patience, delta=0.0):
        """
        Initialize the EarlyStopping instance.
        
        Args:
            patience (int): Number of epochs to wait before stopping.
            delta (float): Minimum improvement in validation loss to reset patience.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')       # Initialize with a very high loss
        self.counter = 0                    # Tracks epochs without improvement
        self.early_stop = False             # Flag to indicate when to stop

    def step(self, val_loss):
        """
        Check if validation loss has improved and update the stopping criteria.
        
        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if val_loss < self.best_loss - self.delta:
            # Update best loss and reset counter if loss improves
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Increment counter if no improvement
            self.counter += 1
            if self.counter >= self.patience:
                # Trigger early stopping if patience is exceeded
                self.early_stop = True

def set_random_seed(seed):
    """
    Set the random seed for reproducibility across all libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False