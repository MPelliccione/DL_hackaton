import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree
import math

class MLPLayer(nn.Module):
    """Multi-layer perceptron layer with batch normalization and dropout"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLPLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class GINPlusLayer(nn.Module):
    """Enhanced GIN layer with edge features and improved aggregation"""
    def __init__(self, in_dim, hidden_dim, edge_dim, dropout=0.1):
        super(GINPlusLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Edge feature transformation
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MLP for aggregation (as per GIN)
        self.mlp = MLPLayer(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        
        # Learnable epsilon parameter
        self.eps = nn.Parameter(torch.zeros(1))
        
        # Attention mechanism for edge-aware aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, edge_index, edge_attr, virtual_node=None):
        # Transform node features
        x_transformed = self.node_encoder(x)
        
        # Transform edge features
        edge_feat = self.edge_encoder(edge_attr)
        
        # Message passing with edge-aware attention
        row, col = edge_index
        
        # Compute attention weights
        node_i = x_transformed[row]  # Source nodes
        node_j = x_transformed[col]  # Target nodes
        
        # Combine node and edge features for attention
        attention_input = torch.cat([node_i + edge_feat, node_j], dim=-1)
        attention_weights = self.attention(attention_input)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted message aggregation
        messages = (node_j + edge_feat) * attention_weights
        
        # Aggregate messages for each node
        out = torch.zeros_like(x_transformed)
        out.index_add_(0, row, messages)
        
        # Add virtual node contribution if provided
        if virtual_node is not None:
            out = out + virtual_node.expand_as(out)
        
        # Apply GIN update rule: (1 + eps) * x + aggregated_messages
        out = (1 + self.eps) * x_transformed + out
        
        # Apply MLP
        out = self.mlp(out)
        
        return out

class VirtualNode(nn.Module):
    """Virtual node implementation for global graph representation"""
    def __init__(self, hidden_dim, dropout=0.1):
        super(VirtualNode, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Virtual node embedding
        self.virtual_emb = nn.Parameter(torch.randn(1, hidden_dim))
        
        # MLPs for virtual node updates
        self.vn_mlp = MLPLayer(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        self.node_mlp = MLPLayer(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        
        # Attention for virtual node interaction
        self.vn_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        
        # Aggregate node features per graph for virtual node update
        vn_input = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Update virtual node representation
        vn_feat = self.virtual_emb.expand(batch_size, -1)
        vn_updated = self.vn_mlp(vn_feat + vn_input)
        
        # Compute attention between virtual node and regular nodes
        vn_expanded = vn_updated[batch]  # Expand to match node dimensions
        attention_input = torch.cat([x, vn_expanded], dim=-1)
        attention_weights = torch.softmax(self.vn_attention(attention_input), dim=0)
        
        # Apply attention to virtual node contribution
        vn_contribution = vn_expanded * attention_weights
        
        return vn_contribution, vn_updated

class NoiseFilter(nn.Module):
    """Noise filtering module for handling noisy labels"""
    def __init__(self, hidden_dim, num_classes, confidence_threshold=0.8):
        super(NoiseFilter, self).__init__()
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Label correction network
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, graph_emb, predictions, labels=None):
        # Estimate confidence in predictions
        confidence = self.confidence_net(graph_emb)
        
        if labels is not None and self.training:
            # During training, identify potentially noisy samples
            pred_probs = F.softmax(predictions, dim=-1)
            max_prob, pred_labels = torch.max(pred_probs, dim=-1)
            
            # Samples with low confidence or disagreement with labels are potentially noisy
            label_agreement = (pred_labels == labels).float().unsqueeze(-1)
            noise_score = 1.0 - (confidence * label_agreement)
            
            # For high noise score samples, attempt label correction
            correction_input = torch.cat([graph_emb, F.one_hot(labels, self.num_classes).float()], dim=-1)
            corrected_logits = self.correction_net(correction_input)
            
            # Blend original and corrected predictions based on noise score
            final_logits = (1 - noise_score) * predictions + noise_score * corrected_logits
            
            return final_logits, confidence, noise_score
        else:
            return predictions, confidence, None

class GINPlusModel(nn.Module):
    """Complete GIN+ model with virtual node and noise filtering"""
    def __init__(self, in_dim=32, edge_dim=32, hidden_dim=300, out_classes=6, 
                 num_layers=5, dropout=0.1, use_virtual_node=True, 
                 confidence_threshold=0.8):
        super(GINPlusModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_virtual_node = use_virtual_node
        self.dropout = dropout
        
        # Input feature transformation
        self.input_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GIN+ layers
        self.gin_layers = nn.ModuleList([
            GINPlusLayer(hidden_dim, hidden_dim, edge_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Virtual node
        if use_virtual_node:
            self.virtual_node = VirtualNode(hidden_dim, dropout)
        
        # Graph-level pooling
        self.pool = nn.ModuleList([
            lambda x, batch: global_add_pool(x, batch),
            lambda x, batch: global_mean_pool(x, batch),
            lambda x, batch: global_max_pool(x, batch)
        ])
        
        # Final classifier
        pooled_dim = hidden_dim * 3  # Concatenation of add, mean, max pooling
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_classes)
        )
        
        # Noise filtering with NCOD principles
        self.noise_filter = NCODNoiseFilter(pooled_dim, out_classes, confidence_threshold)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_attr, batch, labels=None):
        # Input encoding
        x = self.input_encoder(x)
        
        # GIN+ layers with virtual node
        vn_contribution = None
        for i, gin_layer in enumerate(self.gin_layers):
            if self.use_virtual_node:
                if i == 0:
                    # Initialize virtual node contribution
                    vn_contribution, _ = self.virtual_node(x, batch)
                else:
                    # Update virtual node
                    vn_contribution, _ = self.virtual_node(x, batch)
            
            x = gin_layer(x, edge_index, edge_attr, vn_contribution)
            
            # Apply residual connection and layer normalization
            if i > 0:
                x = x + x_prev
            x_prev = x
        
        # Graph-level representation
        graph_emb_list = []
        for pool_fn in self.pool:
            graph_emb_list.append(pool_fn(x, batch))
        
        graph_emb = torch.cat(graph_emb_list, dim=-1)
        
        # Classification
        logits = self.classifier(graph_emb)
        
        # Noise filtering
        filtered_logits, confidence, noise_score = self.noise_filter(graph_emb, logits, labels)
        
        return {
            'logits': filtered_logits,
            'confidence': confidence,
            'noise_score': noise_score,
            'graph_embedding': graph_emb
        }
    
    def get_embeddings(self, x, edge_index, edge_attr, batch):
        """Get graph embeddings without classification"""
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_attr, batch)
            return output['graph_embedding']

# NCOD Loss Implementation
class NCODLoss(nn.Module):
    """
    Noisy Correspondence with Orthogonal Disentanglement Loss
    Based on: https://github.com/wanifarooq/NCOD/blob/main/NCOD.py
    """
    def __init__(self, num_classes, lambda_reg=0.1, lambda_orth=0.01, 
                 temperature=0.5, warmup_epochs=10):
        super(NCODLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg      # Regularization weight
        self.lambda_orth = lambda_orth    # Orthogonality constraint weight
        self.temperature = temperature    # Temperature for softmax
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        # Moving averages for clean sample identification
        self.register_buffer('loss_moving_avg', torch.zeros(1))
        self.register_buffer('update_count', torch.zeros(1))
        
    def set_epoch(self, epoch):
        """Update current epoch for warmup scheduling"""
        self.current_epoch = epoch
    
    def orthogonal_regularization(self, embeddings):
        """
        Compute orthogonal regularization to encourage disentangled representations
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute correlation matrix
        correlation_matrix = torch.mm(embeddings_norm.t(), embeddings_norm)
        
        # Identity matrix
        identity = torch.eye(embeddings_norm.size(1), device=embeddings.device)
        
        # Orthogonal loss (encourage off-diagonal elements to be zero)
        orth_loss = torch.norm(correlation_matrix - identity, p='fro') ** 2
        
        return orth_loss
    
    def compute_sample_weights(self, losses, confidence_scores):
        """
        Compute sample weights based on loss and confidence
        """
        # Update moving average of losses
        current_avg_loss = torch.mean(losses)
        if self.update_count == 0:
            self.loss_moving_avg = current_avg_loss
        else:
            momentum = 0.9
            self.loss_moving_avg = momentum * self.loss_moving_avg + (1 - momentum) * current_avg_loss
        
        self.update_count += 1
        
        # Identify clean samples (low loss relative to moving average)
        loss_threshold = self.loss_moving_avg * 1.5
        clean_mask = (losses < loss_threshold).float()
        
        # Combine with confidence scores
        if confidence_scores is not None:
            confidence_weights = confidence_scores.squeeze()
            sample_weights = clean_mask * confidence_weights + (1 - clean_mask) * 0.1
        else:
            sample_weights = clean_mask + (1 - clean_mask) * 0.1
        
        return sample_weights, clean_mask
    
    def consistency_regularization(self, embeddings, labels, clean_mask):
        """
        Consistency regularization for clean samples
        """
        if torch.sum(clean_mask) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get clean samples
        clean_embeddings = embeddings[clean_mask.bool()]
        clean_labels = labels[clean_mask.bool()]
        
        # Compute pairwise distances for same class samples
        consistency_loss = 0.0
        for class_idx in range(self.num_classes):
            class_mask = (clean_labels == class_idx)
            if torch.sum(class_mask) < 2:
                continue
                
            class_embeddings = clean_embeddings[class_mask]
            
            # Pairwise distances within class (should be small)
            pairwise_dist = torch.cdist(class_embeddings, class_embeddings, p=2)
            
            # Exclude diagonal (distance to self)
            mask = ~torch.eye(pairwise_dist.size(0), dtype=torch.bool, device=pairwise_dist.device)
            intra_class_loss = torch.mean(pairwise_dist[mask])
            
            consistency_loss += intra_class_loss
        
        return consistency_loss / self.num_classes
    
    def adaptive_threshold_selection(self, logits, labels):
        """
        Adaptive threshold selection for noise detection
        """
        with torch.no_grad():
            probs = F.softmax(logits / self.temperature, dim=1)
            pred_probs, predicted = torch.max(probs, dim=1)
            
            # Agreement between prediction and label
            agreement = (predicted == labels).float()
            
            # Dynamic threshold based on current batch statistics
            threshold = torch.quantile(pred_probs, 0.5)  # Median as threshold
            
            # High confidence samples that agree with labels are likely clean
            clean_candidates = (pred_probs > threshold) & (agreement == 1)
            
        return clean_candidates.float()
    
    def forward(self, outputs, labels):
        """
        Forward pass of NCOD loss
        Args:
            outputs: Dictionary containing model outputs
            labels: Ground truth labels (potentially noisy)
        """
        logits = outputs['logits']
        confidence = outputs.get('confidence', None)
        embeddings = outputs['graph_embedding']
        batch_size = logits.size(0)
        
        # Basic cross-entropy loss
        ce_losses = self.ce_loss(logits, labels)
        
        # Adaptive clean sample detection
        clean_mask = self.adaptive_threshold_selection(logits, labels)
        
        # Compute sample weights
        sample_weights, _ = self.compute_sample_weights(ce_losses, confidence)
        
        # Weighted classification loss
        weighted_ce_loss = torch.mean(ce_losses * sample_weights)
        
        # Orthogonal regularization
        orth_loss = self.orthogonal_regularization(embeddings)
        
        # Consistency regularization
        consistency_loss = self.consistency_regularization(embeddings, labels, clean_mask)
        
        # Warmup scheduling
        if self.current_epoch < self.warmup_epochs:
            # During warmup, focus on basic classification
            warmup_factor = self.current_epoch / self.warmup_epochs
            reg_weight = self.lambda_reg * warmup_factor
            orth_weight = self.lambda_orth * warmup_factor
        else:
            reg_weight = self.lambda_reg
            orth_weight = self.lambda_orth
        
        # Total loss
        total_loss = (weighted_ce_loss + 
                     reg_weight * consistency_loss + 
                     orth_weight * orth_loss)
        
        # Return detailed loss information
        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': torch.mean(ce_losses),
            'weighted_ce_loss': weighted_ce_loss,
            'consistency_loss': consistency_loss,
            'orthogonal_loss': orth_loss,
            'clean_ratio': torch.mean(clean_mask),
            'avg_sample_weight': torch.mean(sample_weights)
        }
        
        return total_loss, loss_dict

# Enhanced Noise Filter with NCOD principles
class NCODNoiseFilter(nn.Module):
    """Enhanced noise filter based on NCOD principles"""
    def __init__(self, hidden_dim, num_classes, confidence_threshold=0.8):
        super(NCODNoiseFilter, self).__init__()
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        # Disentangled representation networks
        self.clean_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.noisy_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Label correction network
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, graph_emb, predictions, labels=None):
        # Disentangled representations
        clean_repr = self.clean_branch(graph_emb)
        noisy_repr = self.noisy_branch(graph_emb)
        
        # Combined representation
        combined_repr = torch.cat([clean_repr, noisy_repr], dim=1)
        
        # Confidence estimation
        confidence = self.confidence_net(combined_repr)
        
        if labels is not None and self.training:
            # Label correction based on clean representation
            corrected_logits = self.correction_net(combined_repr)
            
            # Adaptively blend original and corrected predictions
            blend_weight = confidence
            final_logits = (1 - blend_weight) * predictions + blend_weight * corrected_logits
            
            # Compute noise score based on representation disentanglement
            noise_score = 1.0 - confidence
            
            return final_logits, confidence, noise_score
        else:
            return predictions, confidence, None

# Example usage
def create_gin_plus_model():
    """Factory function to create GIN+ model with specified parameters"""
    model = GINPlusModel(
        in_dim=32,
        edge_dim=32,
        hidden_dim=300,
        out_classes=6,
        num_layers=5,
        dropout=0.1,
        use_virtual_node=True,
        confidence_threshold=0.8
    )
    
    return model

# Training utilities with NCOD
def train_step_ncod(model, data, labels, optimizer, criterion, epoch):
    """Single training step with NCOD loss"""
    model.train()
    optimizer.zero_grad()
    
    # Set current epoch for loss scheduling
    criterion.set_epoch(epoch)
    
    outputs = model(data.x, data.edge_index, data.edge_attr, data.batch, labels)
    loss, loss_dict = criterion(outputs, labels)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), loss_dict

def create_ncod_criterion(num_classes=6):
    """Create NCOD loss criterion"""
    return NCODLoss(
        num_classes=num_classes,
        lambda_reg=0.1,      # Consistency regularization weight
        lambda_orth=0.01,    # Orthogonal regularization weight
        temperature=0.5,     # Temperature for softmax
        warmup_epochs=10     # Warmup period
    )

def evaluate(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
            _, predicted = torch.max(outputs['logits'], 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    
    return correct / total

if __name__ == "__main__":
    # Example model instantiation
    model = create_gin_plus_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example NCOD loss function
    criterion = create_ncod_criterion(num_classes=6)
    print("GIN+ model with NCOD loss and virtual node ready!")
    
    # Example training loop setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print("Training setup complete!")
    print("Key features:")
    print("- NCOD loss for robust noisy label handling")
    print("- Orthogonal disentanglement regularization")
    print("- Adaptive clean sample identification")
    print("- Consistency regularization")
    print("- Dynamic sample weighting")