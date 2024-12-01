import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_config

# Base class for 2D shapes
class Shape2D:
    """
    Represents a generic 2D shape, providing basic attributes and transformations like rotation and scaling.
    
    Attributes:
        label (str): The name/type of the shape (e.g., triangle, rectangle).
        nodes (list): Coordinates of the shape's vertices.
        edges (list): Connectivity between nodes defining edges.
    """
    def __init__(self, label):
        self.label = label  # Label to identify the shape type
        self.nodes = []     # List of 2D coordinates for shape vertices
        self.edges = []     # List of edges as pairs of node indices

    def rotate(self, angle):
        """
        Rotate the shape by a given angle in radians.
        
        Args:
            angle (float): Rotation angle in radians.
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.nodes = np.dot(self.nodes, rotation_matrix.T)  # Apply rotation matrix to nodes

    def scale(self, factor):
        """
        Scale the shape by a given factor.
        
        Args:
            factor (float): Scaling factor.
        """
        self.nodes = np.array(self.nodes) * factor  # Scale node coordinates

# Derived classes for specific shapes
class Triangle(Shape2D):
    def __init__(self):
        super().__init__("triangle")
        self._generate()

    def _generate(self):
        """
        Generate a random triangle by creating three random 2D points.
        """
        points = np.random.rand(3, 2)  # Generate 3 random points
        self.nodes = points.tolist()
        self.edges = [(0, 1), (1, 2), (2, 0)]  # Define edges for a triangle

class Rectangle(Shape2D):
    def __init__(self):
        super().__init__("rectangle")
        self._generate()

    def _generate(self):
        """
        Generate a random rectangle defined by a random width and height.
        """
        x, y = np.random.rand(2)  # Bottom-left corner
        width, height = np.random.rand(2) * 0.5  # Random width and height
        self.nodes = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Edges for a rectangle

class Circle(Shape2D):
    def __init__(self):
        super().__init__("circle")
        self._generate()

    def _generate(self):
        """
        Generate a circle represented by points evenly spaced along its perimeter.
        """
        num_points = 30  # Number of points to represent the circle
        angle = np.linspace(0, 2 * np.pi, num_points)
        radius = np.random.rand() * 0.5 + 0.5  # Random radius
        self.nodes = [[np.cos(a) * radius, np.sin(a) * radius] for a in angle]
        self.edges = [(i, (i + 1) % num_points) for i in range(num_points)]  # Circular connectivity

class Hexagon(Shape2D):
    def __init__(self):
        super().__init__("hexagon")
        self._generate()

    def _generate(self):
        """
        Generate a hexagon by calculating six evenly spaced points on a circle.
        """
        angle = np.linspace(0, 2 * np.pi, 7)[:-1]  # Six points
        radius = np.random.rand() * 0.5 + 0.5  # Random radius
        self.nodes = [[np.cos(a) * radius, np.sin(a) * radius] for a in angle]
        self.edges = [(i, (i + 1) % 6) for i in range(6)]  # Hexagon connectivity

class Ellipse(Shape2D):
    def __init__(self):
        super().__init__("ellipse")
        self._generate()

    def _generate(self):
        """
        Generate an ellipse by scaling points from the unit circle.
        """
        num_points = 30  # Number of points to represent the ellipse
        angles = np.linspace(0, 2 * np.pi, num_points)
        semi_major_axis = np.random.rand() * 0.5 + 0.5  # Random semi-major axis
        semi_minor_axis = np.random.rand() * 0.5 + 0.5  # Random semi-minor axis
        self.nodes = [[np.cos(angle) * semi_major_axis, np.sin(angle) * semi_minor_axis] for angle in angles]
        self.edges = [(i, (i + 1) % num_points) for i in range(num_points)]  # Connect points to form an ellipse

# Dataset class for managing shapes
class ShapeDataset:
    def __init__(self):
        config = load_config()
        self.num_samples = config["data"]["num_samples"]  # Number of samples to generate
        self.augment_config = config["augmentation"]      # Augmentation settings
        self.shapes = []                                  # List of generated shapes
        
        # Create directories for saving shapes
        self.train_dir = "shapes/train"
        self.val_dir = "shapes/val"
        self.test_dir = "shapes/test"
        
        # Ensure the directories exist
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
    def generate(self):
        """
        Generate a dataset of random shapes with optional augmentations.
        """
        shape_classes = [Triangle, Rectangle, Circle, Hexagon, Ellipse]
        for idx in range(self.num_samples):
            shape = random.choice(shape_classes)()  # Randomly choose a shape class

            # Apply random rotation if enabled
            if self.augment_config.get("rotation", False):
                angle = random.uniform(0, 2 * np.pi)  # Random angle
                shape.rotate(angle)

            # Apply random scaling if enabled
            if self.augment_config.get("scaling", False):
                scale = random.uniform(
                    self.augment_config.get("min_scale", 0.5),
                    self.augment_config.get("max_scale", 1.5),
                )
                shape.scale(scale)

            self.shapes.append(shape)
            
            # Save shape to appropriate folder (train/val/test)
            split = self._get_split(idx)
            self.save_shape(shape, split, idx)

    def _get_split(self, idx):
        """Return 'train', 'val', or 'test' based on index."""
        if idx < self.num_samples * 0.7:
            return "train"
        elif idx < self.num_samples * 0.85:
            return "val"
        else:
            return "test"

    def save_shape(self, shape, split, idx):
        """Save the shape to a subfolder."""
        # Convert nodes (NumPy array) to list and ensure the edges are also in a serializable format
        shape_data = {
            "nodes": shape.nodes.tolist(),  # Convert NumPy array to list
            "edges": shape.edges  # Edges are already a list of tuples, so no conversion needed
        }
        
        # Only save the label for training data
        if split == "train":
            shape_data["label"] = shape.label

        file_name = f"{shape.label}_{idx}.json" if split == "train" else f"{idx}.json"
        file_path = os.path.join("shapes", split, file_name)
    
        # Save the shape data as JSON
        with open(file_path, 'w') as f:
            json.dump(shape_data, f, indent=4)
    
    def visualize_each_class(self):
        """
        Visualize one example from each shape class.
        """
        shape_classes = [Triangle, Rectangle, Circle, Hexagon, Ellipse]
        for shape_class in shape_classes:
            shape = shape_class()  # Create an instance of the shape
            self._plot_shape(shape)  # Plot the shape

    def _plot_shape(self, shape):
        """
        Plot a single shape, showing its nodes and edges.
        
        Args:
            shape (Shape2D): The shape to visualize.
        """
        nodes = np.array(shape.nodes)
        edges = shape.edges
        plt.figure()
        for start, end in edges:
            plt.plot(
                [nodes[start][0], nodes[end][0]],
                [nodes[start][1], nodes[end][1]],
                "k-"
            )
        plt.scatter(nodes[:, 0], nodes[:, 1], color="red")  # Mark vertices
        plt.title(shape.label)  # Show shape label
        plt.axis("equal")  # Keep aspect ratio
        plt.pause(0.5)
        plt.close()

if __name__ == "__main__":
    # Load configuration
    dataset = ShapeDataset()
    print("Visualizing one sample from each shape class...")
    dataset.visualize_each_class()  # Visualize a sample from each shape class