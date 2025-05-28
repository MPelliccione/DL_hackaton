import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj

class gen_node_features(object):
    # Keep existing implementation
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim
    def __call__(self, data):
        # generate node features if not exist
        if not hasattr(data, 'x') or data.x is None:
            num_nodes = 0 
            if hasattr(data, 'num_nodes') and data.num_nodes is not None:
                num_nodes = data.num_nodes
            elif hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
                num_nodes = data.edge_index.max().item() + 1
            
            if num_nodes > 0:
                data.x = torch.randn((num_nodes, self.feat_dim), dtype=torch.float)
            else:
                data.x = torch.empty((0, self.feat_dim), dtype=torch.float)
                print(f"Warning: Graph has no nodes or edges. Initializing data.x with an empty tensor for graph with y={data.y if hasattr(data, 'y') else 'N/A'}.")

        data.x = torch.nan_to_num(data.x, nan=0.0)
        return data

class Edge2NodeFeatures(nn.Module):
    # Keep existing implementation
    def __init__(self, edge_feat_dim, node_feat_dim):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.LayerNorm(node_feat_dim)
        )
    
    def forward(self, edge_attr, edge_index, num_nodes):
        transformed_edge_features = self.edge_transform(edge_attr)
        node_features = torch.zeros((num_nodes, transformed_edge_features.size(1)), 
                                  device=edge_attr.device)
        node_features.index_add_(0, edge_index[1], transformed_edge_features)
        node_features.index_add_(0, edge_index[0], transformed_edge_features)
        degree = torch.zeros(num_nodes, device=edge_attr.device)
        degree.index_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        degree.index_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=torch.float))
        degree = degree.clamp(min=1).unsqueeze(1)
        return node_features / degree

class GatedGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, edge_feat_dim):
        super().__init__()
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
        self.C = nn.Linear(in_dim, out_dim)
        self.D = nn.Linear(in_dim, out_dim)
        self.E = nn.Linear(edge_feat_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        Ax = self.A(x)
        Bx = self.B(x)
        Cx = self.C(x)
        Dx = self.D(x)
        Ex = self.E(edge_attr)
        
        # Edge attention with edge features
        e_ij = Bx[src] + Cx[dst] + Ex
        sigma_ij = torch.sigmoid(e_ij)
        
        # Message passing
        msg = Ax[src] * sigma_ij
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        
        # Gating and residual
        gated = torch.sigmoid(Dx)
        out = self.bn(out * gated + x)
        return self.dropout(F.relu(out))

class GatedGCNPlus(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_feat_dim, out_classes, n_layers=3):
        super().__init__()
        self.edge2node = Edge2NodeFeatures(edge_feat_dim, in_dim)
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        # GatedGCN layers
        self.layers = nn.ModuleList([
            GatedGCNConv(hidden_dim, hidden_dim, edge_feat_dim)
            for _ in range(n_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_classes)
        )
        
        # Pretraining decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output scalar for each edge
        )
        
    def forward(self, data, enable_classifier=True):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if (x is None) or (edge_attr is None):
            raise ValueError("None values for features data.x or for data.edge_attr!")
        
        # Edge2node transformation
        edge_features = self.edge2node(edge_attr, edge_index, x.size(0))
        x = x + edge_features
        
        # Initial embedding
        x = self.embedding(x)
        
        # GatedGCN+ layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Edge prediction for pretraining
        src, dst = edge_index
        edge_pred = self.decoder(torch.abs(x[src] - x[dst]))
        adj_pred = torch.sigmoid(edge_pred).squeeze()
        
        # Classification
        if enable_classifier:
            graph_embedding = global_mean_pool(x, batch)
            class_logits = self.classifier(graph_embedding)
        else:
            class_logits = None
        
        # Match old interface: (adj_pred, mu, logvar, class_logits, node_embedding)
        return adj_pred, None, None, class_logits, x

