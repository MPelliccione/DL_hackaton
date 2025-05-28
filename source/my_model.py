import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

# node features gen class - mi piaceva metterla qui anche se è più "data preparation"
class gen_node_features(object):
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim
    def __call__(self, data):
        # generate node features if not exist
        if not hasattr(data, 'x') or data.x is None:
            num_nodes = 0 # Default a 0 nodi
            if hasattr(data, 'num_nodes') and data.num_nodes is not None:
                num_nodes = data.num_nodes
            elif hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
                num_nodes = data.edge_index.max().item() + 1
            
            if num_nodes > 0:
                data.x = torch.randn((num_nodes, self.feat_dim), dtype=torch.float)
            else:
                # Se non ci sono nodi validi o edge_index, crea un tensore vuoto per data.x
                data.x = torch.empty((0, self.feat_dim), dtype=torch.float)
                print(f"Warning: Graph has no nodes or edges. Initializing data.x with an empty tensor for graph with y={data.y if hasattr(data, 'y') else 'N/A'}.")

        data.x = torch.nan_to_num(data.x, nan=0.0)
        return data

# Encoder class
class VGAE_encoder(nn.Module):   #- DA FARE CHECK 
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=32):
        super().__init__()
        # Add edge2node transformation
        self.edge2node = Edge2NodeFeatures(edge_feat_dim, in_dim)
        
        nn1_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, in_dim*hid_dim)
        )
        self.conv1 = NNConv(in_dim, hid_dim, nn1_edge_maps, aggr='mean')
        nn_mu_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, hid_dim*lat_dim) 
        )
        self.conv_mu = NNConv(hid_dim,lat_dim, nn_mu_edge_maps, aggr='sum') # test aggr='mean'
        nn_logvar_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, hid_dim*lat_dim)
        )
        self.conv_logvar = NNConv(hid_dim,lat_dim,nn_logvar_edge_maps, aggr='sum') # test aggr='mean'
        self.dropout = nn.Dropout(0.2) # 20% dropout
        
    def forward(self, x, edge_index, edge_attr):
        # Get edge2node features
        edge2node_features = self.edge2node(edge_attr, edge_index, x.size(0))
        
        # Combine original node features with edge2node features
        x = x + edge2node_features
        
        # Continue with regular forward pass
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        # compute node level mean and logvar
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        return mu, logvar

# Decoder class
class VGAE_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# final class all the model here   - new release: from CGNConv to NNConv
class VGAE_all(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=32, 
                 out_classes=6, hid_dim_classifier=64):
        super().__init__()
        self.encoder = VGAE_encoder(in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim)
        self.decoder = VGAE_decoder()
        self.classifier = nn.Sequential(
            nn.Linear(lat_dim, hid_dim_classifier),
            nn.ReLU(),
            # add a 10% dropout to avoid/mitigate overfitting - try diff values 
            nn.Dropout(0.1),
            nn.Linear(hid_dim_classifier, out_classes)
        )
                     
    #  maybe possiamo inserire qui la parte di concatenazione in decoder invece di goto_the_gym.py
    def forward(self, data, enable_classifier=True):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # check on x and edge_attr != None
        if (x is None) or (edge_attr is None):
            raise ValueError("None values for features data.x or for data.edge_attr!")
            
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = reparametrize(mu, logvar)
        adj_pred = self.decoder(z) if z is not None else None              

        # pooling if classifier was enabled: in pre-training we work only with VGAE
        if enable_classifier:
            if z is None:
                raise ValueError("Latent node embeddings 'z' are None! Unable to use the classifier.")
            graph_embedding = global_mean_pool(z, batch)
            class_logits = self.classifier(graph_embedding)
        # else if classifier is NOT enabled then class_logits = None
        else:
            class_logits = None
            
        return adj_pred, mu, logvar, class_logits, z

class Edge2NodeFeatures(nn.Module):
    def __init__(self, edge_feat_dim, node_feat_dim):
        super().__init__()
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.LayerNorm(node_feat_dim)
        )
        
    def forward(self, edge_attr, edge_index, num_nodes):
        # Transform edge features
        transformed_edge_features = self.edge_transform(edge_attr)
        
        # Aggregate edge features for each node
        node_features = torch.zeros((num_nodes, transformed_edge_features.size(1)), 
                                  device=edge_attr.device)
        
        # Add incoming edge features to nodes
        node_features.index_add_(0, edge_index[1], transformed_edge_features)
        # Add outgoing edge features to nodes
        node_features.index_add_(0, edge_index[0], transformed_edge_features)
        
        # Normalize by degree (number of edges per node)
        degree = torch.zeros(num_nodes, device=edge_attr.device)
        degree.index_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        degree.index_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=torch.float))
        degree = degree.clamp(min=1).unsqueeze(1)
        
        return node_features / degree

