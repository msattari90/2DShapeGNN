import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_config

class Shape2D:
    """
    Base class for all 2D shapes.
    Contains methods for rotation and scaling, which can be applied to shapes.
    """
    def __init__(self, label):
        self.label = label
        self.nodes = []
        self.edges = []

    def rotate(self, angle):
        """Rotate the shape by a given angle (in radians)."""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.nodes = np.dot(self.nodes, rotation_matrix.T)

    def scale(self, factor):
        """Scale the shape by a given factor."""
        self.nodes = np.array(self.nodes) * factor

# Shape classes for Triangle, Rectangle, Circle, Hexagon, Ellipse...
class Triangle(Shape2D):
    def __init__(self):
        super().__init__("triangle")
        self._generate()

    def _generate(self):
        points = np.random.rand(3, 2)
        self.nodes = points.tolist()
        self.edges = [(0, 1), (1, 2), (2, 0)]

class Rectangle(Shape2D):
    def __init__(self):
        super().__init__("rectangle")
        self._generate()

    def _generate(self):
        x, y = np.random.rand(2)
        width, height = np.random.rand(2) * 0.5
        self.nodes = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

# Shape Dataset
class ShapeDataset:
    def __init__(self):
        config = load_config()
        self.num_samples = config["data"]["num_samples"]
        self.seed = config["data"]["seed"]
        self.rng = np.random.default_rng(self.seed)
        self.shapes = []

    def generate(self):
        """Generate random shapes with optional augmentations."""
        for _ in range(self.num_samples):
            shape_type = random.choice([Triangle, Rectangle, Circle, Hexagon, Ellipse])
            shape = shape_type()

            # Apply augmentations based on config
            config = load_config()
            if config["augmentation"]["rotation"] and random.random() > 0.5:
                shape.rotate(random.uniform(0, 2 * np.pi))
            if config["augmentation"]["scaling"] and random.random() > 0.5:
                scale_factor = random.uniform(config["augmentation"]["min_scale"], config["augmentation"]["max_scale"])
                shape.scale(scale_factor)

            self.shapes.append(shape)

    def visualize(self):
        """Visualize the generated shapes."""
        for shape in self.shapes[:10]:
            self._plot_shape(shape)

    def _plot_shape(self, shape):
        """Helper function to plot a single shape."""
        nodes = np.array(shape.nodes)
        edges = shape.edges
        plt.figure()
        for edge in edges:
            start, end = edge
            plt.plot([nodes[start, 0], nodes[end, 0]], [nodes[start, 1], nodes[end, 1]], 'k-')
        plt.scatter(nodes[:, 0], nodes[:, 1], color='red')
        plt.title(shape.label)
        plt.show()

