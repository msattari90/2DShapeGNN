import tkinter as tk
from tkinter import filedialog
import json
import torch
import torch_geometric
from torch_geometric.data import Data
from GNNModel import SimpleGNN
from utils import load_config
import matplotlib.pyplot as plt
import numpy as np

class ShapePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape Prediction")
        self.root.geometry("400x300")
        
        # Add button to load shape file
        self.load_button = tk.Button(self.root, text="Load Shape File", command=self.load_shape_file)
        self.load_button.pack(pady=20)
        
        # Label to show prediction result
        self.result_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 14))
        self.result_label.pack(pady=20)

        # Load trained model
        config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGNN(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            output_dim=config["model"]["output_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"]
        ).to(self.device)
        self.model.load_state_dict(torch.load("trained_model.pth"))
        self.model.eval()
        
        # Define label mapping from string to numeric values
        self.label_mapping = {
            "triangle": 0,
            "rectangle": 1,
            "circle": 2,
            "hexagon": 3,
            "ellipse": 4
        }

    def load_shape_file(self):
        """Allow the user to select a shape file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.predict_label(file_path)

    def predict_label(self, file_path):
        """Load shape data, preprocess, and predict label."""
        with open(file_path, "r") as f:
            shape_data = json.load(f)

        # Convert shape data into graph format for GNN
        nodes = torch.tensor(shape_data["nodes"], dtype=torch.float).to(self.device)
        edges = torch.tensor(shape_data["edges"], dtype=torch.long).t().contiguous().to(self.device)
        
        # Convert string label to numeric using the label mapping
        label_str = shape_data["label"]
        label_numeric = self.label_mapping[label_str]
        
        # Create a graph data object
        data = Data(x=nodes, edge_index=edges, y=torch.tensor([label_numeric], dtype=torch.long).to(self.device))

        # Predict the label
        with torch.no_grad():
            output = self.model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long).to(self.device))
            predicted_label = output.argmax(dim=1).item()

            # Find the predicted shape label
            predicted_shape = list(self.label_mapping.keys())[list(self.label_mapping.values()).index(predicted_label)]

        # Update the label in the GUI
        self.result_label.config(text=f"Prediction: {predicted_shape}")
        
        # Plot the shape
        self.plot_shape(shape_data)

    def plot_shape(self, shape_data):
        """Plot the selected shape."""
        nodes = np.array(shape_data["nodes"])
        edges = shape_data["edges"]

        plt.figure(figsize=(6, 6))
        for start, end in edges:
            plt.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]], "k-")
        plt.scatter(nodes[:, 0], nodes[:, 1], color="red")
        plt.title(f"Shape: {shape_data['label']}")
        plt.axis("equal")
        plt.show()
        
# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ShapePredictionApp(root)
    root.mainloop()
