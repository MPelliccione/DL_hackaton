import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, negative_sampling
from torch.cuda.amp import autocast, GradScaler
from my_model import GatedGCNPlus
import math
import numpy as np
""" # our beloved Kullback-Leibler term loss
def kl_loss(mu, logvar):
    # clip logvar to avoid extreme values 
    clip_logvar = torch.clamp(logvar, min=-5.0, max=5.0) 
    return -0.5 * torch.mean(1 + clip_logvar -mu.pow(2) - clip_logvar.exp())
"""
# reconstruction loss function
def eval_reconstruction_loss(adj_pred, edge_index, num_nodes, num_neg_samp=1):
    eps = 1e-7
    
    # Ensure adj_pred is on correct device and has valid values
    adj_pred = torch.clamp(adj_pred, min=eps, max=1.0-eps)
    
    try:
        # Sample only a subset of negative edges for memory efficiency
        max_neg_samples = min(edge_index.size(1) * num_neg_samp, 10000)
        
        positive_logits = adj_pred[edge_index[0], edge_index[1]]
        positive_labels = torch.ones_like(positive_logits)

        neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=max_neg_samples)
            
        negative_logits = adj_pred[neg_edge_index[0], neg_edge_index[1]]
        negative_labels = torch.zeros_like(negative_logits)

        all_logits = torch.cat([positive_logits, negative_logits])
        all_labels = torch.cat([positive_labels, negative_labels])
        
        # Safety clamp
        all_logits = torch.clamp(all_logits, min=eps, max=1.0-eps)
        
        recon_loss = F.binary_cross_entropy(all_logits, all_labels)
        return recon_loss

    except Exception as e:
        print(f"Error in reconstruction loss: {e}")
        return torch.tensor(0.0, device=adj_pred.device)

class NCODLoss(nn.Module):
    def __init__(self, labels, n_samples, num_classes, ratio_consistency=0, ratio_balance=0):
        super(NCODLoss, self).__init__()
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance
        
        # Inizializza u come parametro learnable
        self.u = nn.Parameter(torch.empty(n_samples, 1, dtype=torch.float32))
        self.init_param()
        
        # Salva labels e crea bins per classe
        self.labels = labels
        self.bins = []
        for i in range(num_classes):
            self.bins.append(np.where(self.labels == i)[0])
            
        # Inizializza embedding memory
        self.beginning = True
        self.prev_phi_x_i = torch.rand((n_samples, 512))  # 512 è la dim dell'embedding
        self.phi_c = torch.rand((num_classes, 512))

    def init_param(self, mean=1e-8, std=1e-9):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        
    def forward(self, index, logits, y_onehot, phi_x_i, flag=0, epoch=0):
        eps = 1e-4
        u = self.u[index].to(logits.device)
        
        # Aggiorna i centroidi di classe
        if flag == 0 and self.beginning:
            percent = math.ceil((50 - (50 / 5) * epoch) + 50)  # 5 epochs totali
            for i in range(len(self.bins)):
                class_u = self.u.detach()[self.bins[i]]
                bottomK = int((len(class_u) / 100) * percent)
                important_indexs = torch.topk(class_u, bottomK, largest=False, dim=0)[1]
                self.phi_c[i] = torch.mean(self.prev_phi_x_i[self.bins[i]][important_indexs.view(-1)], dim=0)

        # Calcola similarità con i centroidi
        phi_c_norm = self.phi_c.norm(p=2, dim=1, keepdim=True)
        h_c_bar = self.phi_c.div(phi_c_norm.clamp(min=eps))
        h_c_bar_T = torch.transpose(h_c_bar, 0, 1)

        # Aggiorna memoria degli embedding
        self.prev_phi_x_i[index] = phi_x_i.detach()

        # Calcola probabilità e loss
        f_x_softmax = F.softmax(logits, dim=1)
        phi_x_i_norm = phi_x_i.detach().norm(p=2, dim=1, keepdim=True)
        h_i = phi_x_i.detach().div(phi_x_i_norm.clamp(min=eps))

        y_bar = torch.mm(h_i, h_c_bar_T)
        y_bar = y_bar * y_onehot
        y_bar_max = (y_bar > 0.000).float()
        y_bar = y_bar * y_bar_max

        u = u * y_onehot
        f_x_softmax = torch.clamp(f_x_softmax + u.detach(), min=eps, max=1.0)
        
        # Loss principale
        L1 = torch.mean(-torch.sum(y_bar * torch.log(f_x_softmax), dim=1))

        # Loss aggiuntive
        y_hat = self.soft_to_hard(logits.detach())
        L2 = F.mse_loss((y_hat + u), y_onehot, reduction='sum') / len(y_onehot)
        
        total_loss = L1 + L2

        # Balance loss opzionale
        if self.ratio_balance > 0:
            avg_prediction = torch.mean(f_x_softmax, dim=0)
            prior_distr = torch.ones_like(avg_prediction) / self.num_classes
            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            total_loss += self.ratio_balance * balance_kl

        return total_loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return torch.zeros(len(x), self.num_classes).to(x.device).scatter_(
                1, torch.argmax(x, dim=1).view(-1, 1), 1
            )

# Pre training procedure - no classifiers here
def pretraining(model, td_loader, optimizer, device, cur_epoch):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    print(f"PRETRAINING: Epoch {cur_epoch + 1}")    
    
    for batch_idx, data in enumerate(td_loader):
        try:
            data = data.to(device)
            optimizer.zero_grad()

            # Get edge reconstruction prediction
            adj_pred, _, _, _, _ = model(data, enable_classifier=False)
            
            if adj_pred is None:
                print(f"Warning: Model output is None in batch {batch_idx}. Skipping")
                continue

            # Compute reconstruction loss
            reconstruction_loss = eval_reconstruction_loss(
                adj_pred, 
                data.edge_index, 
                data.x.size(0), 
                num_neg_samp=1
            )
            
            if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                print(f"Warning: Invalid loss in batch {batch_idx}. Skipping")
                continue

            # Backprop
            reconstruction_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += reconstruction_loss.item()
            valid_batches += 1

        except RuntimeError as e:
            print(f"Runtime error in batch {batch_idx}: {e}")
            continue
            
    return total_loss/max(valid_batches, 1)
    
# Training procedure - classifier is in!
def train(model, td_loader, optimizer, device, cur_epoch):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    for batch_idx, data in enumerate(td_loader):
        try:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Get predictions with classifier enabled
            class_logits, _ = model(data)
            
            if class_logits is None:
                print(f"Warning: Model output is None in batch {batch_idx}. Skipping")
                continue
            
            # Classification loss
            loss = F.cross_entropy(class_logits, data.y)
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1

        except RuntimeError as e:
            print(f"Runtime error in batch {batch_idx}: {e}")
            continue
            
    return total_loss/max(valid_batches, 1)
