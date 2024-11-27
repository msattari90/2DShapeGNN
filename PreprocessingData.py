import torch
from torch_geometric.data import Data
from utils import load_config

class GraphPreprocessor:
    def __init__(self, shapes):
        self.shapes = shapes
        self.graphs = []

    def preprocess(self):
        """Convert each shape into a graph."""
        for shape in self.shapes:
            graph = self._convert_to_graph(shape)
            self.graphs.append(graph)

    def _convert_to_graph(self, shape):
        """Convert a shape to graph format with nodes, edges, and labels."""
        nodes = torch.tensor(shape.nodes, dtype=torch.float)
        edges = self._build_edge_index(shape.edges)
        label = self._encode_label(shape.label)

        return Data(x=nodes, edge_index=edges, y=label)

    def _build_edge_index(self, edges):
        """Build the edge index for the graph."""
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _encode_label(self, label):
        """Encode shape labels as integers."""
        label_mapping = {"triangle": 0, "rectangle": 1, "circle": 2, "hexagon": 3, "ellipse": 4}
        return torch.tensor([label_mapping[label]], dtype=torch.long)

    def get_graphs(self):
        """Return the list of preprocessed graphs."""
        return self.graphs
