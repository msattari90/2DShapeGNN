import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_config

class Shape2D:
    """
    Base class for all 2D shapes with rotation and scaling augmentation.
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

class Circle(Shape2D):
    def __init__(self):
        super().__init__("circle")
        self._generate()

    def _generate(self):
        # Generate a circle by using points on the unit circle, scaled to random size
        num_points = 30  # Number of points to represent the circle
        angle = np.linspace(0, 2 * np.pi, num_points)
        radius = np.random.rand() * 0.5 + 0.5  # Random radius between 0.5 and 1
        self.nodes = [[np.cos(a) * radius, np.sin(a) * radius] for a in angle]
        self.edges = [(i, (i+1) % num_points) for i in range(num_points)]  # Connect points to form a circle

class Hexagon(Shape2D):
    def __init__(self):
        super().__init__("hexagon")
        self._generate()

    def _generate(self):
        # Generate hexagon by calculating 6 points spaced evenly around a circle
        angle = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points in a circle (remove last to avoid duplicate point)
        radius = np.random.rand() * 0.5 + 0.5  # Random radius between 0.5 and 1
        self.nodes = [[np.cos(a) * radius, np.sin(a) * radius] for a in angle]
        self.edges = [(i, (i+1) % 6) for i in range(6)]  # Connect points to form a hexagon

class Ellipse(Shape2D):
    def __init__(self):
        super().__init__("ellipse")
        self._generate()

    def _generate(self):
        # Generate ellipse by scaling points from the unit circle
        num_points = 30
        angle = np.linspace(0, 2 * np.pi, num_points)
        a = np.random.rand() * 0.5 + 0.5  # Random semi-major axis between 0.5 and 1
        b = np.random.rand() * 0.5 + 0.5  # Random semi-minor axis between 0.5 and 1
        self.nodes = [[np.cos(a) * a, np.sin(a) * b] for a in angle]
        self.edges = [(i, (i+1) % num_points) for i in range(num_points)]  # Connect points to form an ellipse

# Shape Dataset
class ShapeDataset:
    def __init__(self):
        config = load_config()
        self.num_samples = config["data"]["num_samples"]
        self.augment_config = config["augmentation"]
        self.shapes = []
        
    def generate(self):
        """Generate random shapes with optional augmentations."""
        shape_classes = [Triangle, Rectangle, Circle, Hexagon, Ellipse]
        for _ in range(self.num_samples):
            shape = random.choice(shape_classes)()
            
            # Apply augmentations if configured
            if self.augment_config.get("rotation", False):
                angle = random.uniform(0, 2 * np.pi)  # Generate a random angle
                shape.rotate(angle)

            if self.augment_config.get("scaling", False):
                scale = random.uniform(
                    self.augment_config.get("min_scale", 0.5),
                    self.augment_config.get("max_scale", 1.5),
                )
                shape.scale(scale)

            self.shapes.append(shape)

    def visualize(self, num_samples=5):
        """Visualize the generated shapes."""
        for shape in self.shapes[:num_samples]:
            self._plot_shape(shape)
    
    def _plot_shape(self, shape):
        """Helper function to plot a single shape."""
        nodes = np.array(shape.nodes)
        edges = shape.edges
        plt.figure()
        for start, end in edges:
            plt.plot(
                [nodes[start][0], nodes[end][0]],
                [nodes[start][1], nodes[end][1]],
                "k-"
            )
        plt.scatter(nodes[:, 0], nodes[:, 1], color="red")
        plt.title(shape.label)
        plt.axis("equal")
        plt.show()