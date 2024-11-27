import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from utils import load_config

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        """
        Initialize the GNN model with dynamic layers.
        - input_dim: Number of input features (e.g., 2 for x and y).
        - hidden_dim: Number of hidden units in each layer.
        - output_dim: Number of output classes (e.g., 5).
        - num_layers: Number of GCN layers.
        - dropout: Dropout probability to prevent overfitting.
        """
        super(SimpleGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Create multiple GCN layers dynamically
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))  # First layer
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Fully connected layer for graph classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the GNN model.
        - x: Node feature matrix.
        - edge_index: Edge index for graph structure.
        - batch: Batch tensor for graph-level classification.
        """
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  # Apply each GCN layer with ReLU activation
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Global pooling (mean) to get graph-level features
        x = global_mean_pool(x, batch)

        # Output layer
        return F.log_softmax(self.fc(x), dim=1)

# Training function
def train_gnn(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index, data.batch)

        # Compute loss
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)
