import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from utils import load_config
from GNNModel import SimpleGNN
from PreprocessingData import GraphPreprocessor
from SyntheticDataGenerationScript import ShapeDataset

class Evaluator:
    """
    Evaluates the trained GNN model on the test set.
    """
    def __init__(self, model, loader, device):
        self.model = model.to(device)
        self.loader = loader
        self.device = device

    def evaluate(self):
        """
        Evaluate the model and return predictions.
        Returns:
            list: A list of tuples (data, predicted_labels).
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                pred = self.model(data.x, data.edge_index, data.batch).argmax(dim=1)
                predictions.append((data, pred))  # Add batch of predictions
        return predictions
    

class Visualizer:
    """
    Visualizes graphs and predictions.
    """
    @staticmethod
    def plot_graph(data, pred, label_mapping):
        """
        Plot a graph with its predicted label.
        Args:
            data (torch_geometric.data.Data): The graph data.
            pred (torch.Tensor): Predicted label for a single graph.
            label_mapping (dict): Mapping of label indices to names.
        """
        nodes = data.x.cpu().numpy()
        edges = data.edge_index.cpu().numpy()

        plt.figure(figsize=(6, 6))
        # Plot edges
        for edge in edges.T:
            start, end = edge
            plt.plot(
                [nodes[start, 0], nodes[end, 0]],
                [nodes[start, 1], nodes[end, 1]],
                color="gray",
                alpha=0.5,
            )
        # Plot nodes
        plt.scatter(nodes[:, 0], nodes[:, 1], color="blue", s=100, alpha=0.8)
        plt.title(f"Predicted: {label_mapping[pred]}")
        plt.pause(1)  # Show the figure for 1 second
        plt.close()  # Close the figure after 1 second

def split_data(graphs, train_split, val_split):
    """
    Splits the dataset into training, validation, and test sets.
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

    return train_graphs, val_graphs, test_graphs


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Load processed graphs
    graphs = torch.load("processed_graphs.pt")
    train_graphs, val_graphs, test_graphs = split_data(
        graphs,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["validation_split"]
    )

    # Create DataLoader for test data
    test_loader = DataLoader(test_graphs, batch_size=config["training"]["batch_size"], shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Load trained model weights
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    # Evaluate the model
    # Evaluate the model
    evaluator = Evaluator(model, test_loader, device)
    predictions = evaluator.evaluate()

    # Visualize predictions
    label_mapping = {0: "triangle", 1: "rectangle", 2: "circle", 3: "hexagon", 4: "ellipse"}
    visualizer = Visualizer()
    for data, preds in predictions:  # Loop over batches
        for graph_idx, pred in enumerate(preds):  # Loop over graphs in the batch
            visualizer.plot_graph(data[graph_idx], pred.item(), label_mapping)