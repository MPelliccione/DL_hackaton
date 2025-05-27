# Directly copied from Deep Learning Hackthon GitHub repository


import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm 
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        super().__init__(None, transform, pre_transform)
        # Load graphs after super() initialization
        self.graphs = self._load_data()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def _load_data(self):
        print(f"Loading graphs from {self.raw}...")
        print("This may take a few minutes, please wait...")
        
        # Try to load the file
        try:
            # First attempt: try as gzip
            try:
                with gzip.open(self.raw, "rt", encoding="utf-8") as f:
                    graphs_dicts = json.load(f)
            except (gzip.BadGzipFile, OSError):
                # Second attempt: try as regular JSON
                with open(self.raw, 'r', encoding='utf-8') as f:
                    graphs_dicts = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load file {self.raw}: {str(e)}")

        # Convert dictionaries to graph objects
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs

def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)