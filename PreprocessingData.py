import torch
from torch_geometric.data import Data
from utils import load_config
from SyntheticDataGenerationScript import ShapeDataset

class GraphPreprocessor:
    """
    Converts shapes into graph representations suitable for Graph Neural Networks (GNNs).
    
    Attributes:
        shapes (list): List of Shape2D objects to preprocess.
        graphs (list): List of graph objects created from the shapes.
    """
    def __init__(self, shapes):
        self.shapes = shapes  # List of Shape2D objects
        self.graphs = []      # List of processed graph objects

    def preprocess(self):
        """
        Convert each shape into a graph and store the results.
        """
        for shape in self.shapes:
            graph = self._convert_to_graph(shape)
            self.graphs.append(graph)

    def _convert_to_graph(self, shape):
        """
        Convert a shape to graph format with nodes, edges, and labels.
        
        Args:
            shape (Shape2D): A shape object to convert.
            
        Returns:
            Data: A PyTorch Geometric Data object representing the graph.
        """
        # Node features: Coordinates of the shape's vertices
        nodes = torch.tensor(shape.nodes, dtype=torch.float)

        # Edge index: Connectivity between nodes
        edges = self._build_edge_index(shape.edges)

        # Graph label: Encoded shape type (triangle, rectangle, etc.)
        label = self._encode_label(shape.label)

        # Create and return a PyTorch Geometric Data object
        return Data(x=nodes, edge_index=edges, y=label)

    def _build_edge_index(self, edges):
        """
        Build the edge index for the graph, representing node connectivity.
        
        Args:
            edges (list): List of edges as pairs of node indices.
            
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges].
        """
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _encode_label(self, label):
        """
        Encode shape labels as integers for classification tasks.
        
        Args:
            label (str): Label of the shape (e.g., "triangle", "rectangle").
            
        Returns:
            torch.Tensor: Encoded label as a tensor.
        """
        label_mapping = {"triangle": 0, "rectangle": 1, "circle": 2, "hexagon": 3, "ellipse": 4}
        return torch.tensor([label_mapping[label]], dtype=torch.long)

    def get_graphs(self):
        """
        Retrieve the list of preprocessed graphs.
        
        Returns:
            list: List of PyTorch Geometric Data objects.
        """
        return self.graphs


if __name__ == "__main__":
    """
    Main script to generate shapes, preprocess them into graphs, and save the results.
    """
    # Step 1: Load configuration and generate shapes
    print("Generating synthetic shapes...")
    dataset = ShapeDataset()
    dataset.generate()

    # Step 2: Preprocess shapes into graph representations
    print("Preprocessing shapes into graphs...")
    preprocessor = GraphPreprocessor(dataset.shapes)
    preprocessor.preprocess()

    # Retrieve the list of graphs
    graphs = preprocessor.get_graphs()

    # Debugging: Ensure graphs are not empty
    print(f"Total graphs generated: {len(graphs)}")
    assert len(graphs) > 0, "Graph preprocessing resulted in an empty dataset!"

    # Step 3: Save preprocessed graphs for future use
    torch.save(graphs, "processed_graphs.pt")
    print("Preprocessed graphs saved to 'processed_graphs.pt'.")
