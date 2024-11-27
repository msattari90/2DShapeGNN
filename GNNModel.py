import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from utils import load_config, EarlyStopping
import random
import numpy as np

# Set random seeds for reproducibility
def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across all libraries.
    
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleGNN(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) for shape classification.
    
    Attributes:
        convs (nn.ModuleList): List of GCN layers for message passing.
        fc (nn.Linear): Fully connected layer for classification.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        """
        Initialize the GNN model.
        
        Args:
            input_dim (int): Number of input features per node.
            hidden_dim (int): Number of hidden units in GCN layers.
            output_dim (int): Number of output classes.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout rate for regularization.
        """
        super(SimpleGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))  # First GCN layer
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # Additional GCN layers
        self.fc = nn.Linear(hidden_dim, output_dim)  # Final fully connected layer
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass for the GNN model.
        
        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge index defining graph connectivity.
            batch (torch.Tensor): Batch tensor mapping nodes to graphs.
            
        Returns:
            torch.Tensor: Log-softmax predictions for graph classification.
        """
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  # Apply GCN layer with ReLU activation
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Global mean pooling to get graph-level features
        x = global_mean_pool(x, batch)

        # Output layer with log-softmax for classification
        return F.log_softmax(self.fc(x), dim=1)

# Training function
def train_gnn(model, train_loader, val_loader, optimizer, criterion, device, early_stopping):
    """
    Train the GNN model with early stopping.
    
    Args:
        model (nn.Module): The GNN model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on (CPU or GPU).
        early_stopping (EarlyStopping): Early stopping object.
        
    Returns:
        bool: Indicates if early stopping was triggered.
    """
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    val_loss = evaluate_gnn(model, val_loader, criterion, device)
    early_stopping.step(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        return total_loss / len(train_loader), True

    return total_loss / len(train_loader), False

# Validation function
def evaluate_gnn(model, loader, criterion, device):
    """
    Evaluate the GNN model on a dataset.
    
    Args:
        model (nn.Module): The GNN model.
        loader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on (CPU or GPU).
        
    Returns:
        float: Average loss on the dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()

    return total_loss / len(loader)

# Testing function
def test_gnn(model, loader, device):
    """
    Test the GNN model and calculate accuracy.
    
    Args:
        model (nn.Module): The GNN model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to test on (CPU or GPU).
        
    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

# Split data into train, validation, and test sets
def split_data(graphs, train_split, val_split):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        graphs (list): List of graphs.
        train_split (float): Fraction of data for training.
        val_split (float): Fraction of data for validation.
        
    Returns:
        tuple: Train, validation, and test datasets.
    """
    num_graphs = len(graphs)
    train_size = int(train_split * num_graphs)
    val_size = int(val_split * num_graphs)

    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]

    # Debugging: Ensure splits are non-empty
    assert len(train_graphs) > 0, "Training dataset is empty!"
    assert len(val_graphs) > 0, "Validation dataset is empty!"
    assert len(test_graphs) > 0, "Test dataset is empty!"

    return train_graphs, val_graphs, test_graphs

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    set_random_seed(config["data"]["seed"])

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed graphs
    graphs = torch.load("processed_graphs.pt")
    train_graphs, val_graphs, test_graphs = split_data(
        graphs,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["validation_split"]
    )

    # Initialize DataLoader
    train_loader = DataLoader(train_graphs, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=config["training"]["batch_size"], shuffle=False)

    # Initialize model, optimizer, and criterion
    model = SimpleGNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"], delta=1e-4)

    # Train the model
    print("Starting training...")
    for epoch in range(config["training"]["num_epochs"]):
        train_loss, stopped_early = train_gnn(
            model, train_loader, val_loader, optimizer, criterion, device, early_stopping
        )
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}, Loss: {train_loss:.4f}")
        if stopped_early:
            break

    # Test the model
    print("Testing the model...")
    test_acc = test_gnn(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved as 'trained_model.pth'.")