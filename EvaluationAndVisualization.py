import torch
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from utils import load_config
from GNNModel import SimpleGNN
from PreprocessingData import GraphPreprocessor
from SyntheticDataGenerationScript import ShapeDataset

# Visualizer class for plotting graphs
class Visualizer:
    @staticmethod
    def plot_graph(data, pred, label_mapping):
        """Plot a graph with predicted label."""
        plt.figure(figsize=(6, 6))
        nodes = data.x.numpy()
        edges = data.edge_index.numpy()

        # Plot nodes
        plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=100, alpha=0.8)

        # Plot edges
        for edge in edges.T:
            start, end = edge
            plt.plot([nodes[start, 0], nodes[end, 0]], [nodes[start, 1], nodes[end, 1]], "gray", alpha=0.5)

        # Title with prediction
        predicted_label = label_mapping[pred[0]]
        plt.title(f"Graph Classification: {predicted_label}")
        plt.axis("equal")
        plt.show()

# Main script to evaluate the model
if __name__ == "__main__":
    config = load_config()

    # Generate and preprocess data
    dataset = ShapeDataset()
    dataset.generate()
    preprocessor = GraphPreprocessor(dataset.shapes)
    preprocessor.preprocess()
    graphs = preprocessor.get_graphs()

    # Split data into training, validation, and test sets
    train_size = int(config["data"]["train_split"] * len(graphs))
    validation_size = int(config["data"]["validation_split"] * len(graphs))
    train_graphs = graphs[:train_size]
    validation_graphs = graphs[train_size:train_size + validation_size]
    test_graphs = graphs[train_size + validation_size:]

    # Create DataLoader
    train_loader = DataLoader(train_graphs, batch_size=config["training"]["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_graphs, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=config["training"]["batch_size"], shuffle=False)

    # Initialize model
    model = SimpleGNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    # Evaluate and visualize predictions
    evaluator = Evaluator(model, test_loader, device)
    predictions = evaluator.evaluate()

    label_mapping = {0: "triangle", 1: "rectangle", 2: "circle", 3: "hexagon", 4: "ellipse"}
    visualizer = Visualizer()

    for data, pred in predictions:
        visualizer.plot_graph(data, pred, label_mapping)
