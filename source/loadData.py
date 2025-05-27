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
        self.num_graphs, self.graphs_dicts = self._count_graphs()
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        
        # Try first as gzip
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                graphs_dicts = json.load(f)
        except gzip.BadGzipFile:
            # If not gzipped, try regular JSON
            with open(path, 'r', encoding='utf-8') as f:
                graphs_dicts = json.load(f)
        
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs

    def _count_graphs(self):
        graphs_dicts = None
        try:
            # First try to open as gzip
            with gzip.open(self.raw, 'rt', encoding='utf-8') as f:  # Changed from self.json_path to self.raw
                graphs_dicts = json.load(f)
        except gzip.BadGzipFile:
            # If not gzipped, try regular JSON
            with open(self.raw, 'r', encoding='utf-8') as f:  # Changed from self.json_path to self.raw
                graphs_dicts = json.load(f)
                
        if graphs_dicts is None:
            raise ValueError(f"Could not load file: {self.raw}")
            
        return len(graphs_dicts), graphs_dicts



def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)