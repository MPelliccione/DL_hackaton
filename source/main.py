import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from loadData import GraphDataset
import pandas as pd 
from goto_the_gym import pretraining, train
from utilities import create_dirs, save_checkpoint, add_zeros
from my_model import VGAE_all, gen_node_features
from sklearn.metrics import f1_score

KAGGLE_DATASET_PATH = "/kaggle/input/ogbg-ppa-dlhackaton"
KAGGLE_OUTPUT_PATH = "/kaggle/working"

def init_weights(m):
    """Initialize network weights using Xavier initialization"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data, enable_classifier=True)
            class_logits = output[3]
            pred = class_logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                true_labels.extend(data.y.cpu().numpy())
    
    if calculate_accuracy:
        accuracy = correct / total
        f1 = f1_score(true_labels, predictions, average='weighted')
        return accuracy, f1, predictions
    return predictions

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print("Edge2Node added")
    
    # create directories
    create_dirs()
    
    
    in_dim = 8          # Increased for richer feature representation
    hid_dim = 128        # Increased for more complex pattern learning
    lat_dim = 16         # Increased for better latent space
    out_classes = 6      # Keep as is (problem specific)
    edge_feat_dim = 7    # Keep as is (problem specific)
    hid_edge_nn_dim = 64 # Increased for better edge processing
    hid_dim_classifier = 64 # Increased for better classification
    
    pretrain_epoches = 3   # More pretraining epochs
    num_epoches = 3        # More training epochs
    learning_rate = 0.001   # Higher initial learning rate
    bas = 32              # Larger batch size
    dropout_rate = 0.3     # Add dropout for regularization
    
    # Remove unused KL parameters
    torch.manual_seed(0)
    
    # Initialize model and optimizer first
    model = VGAE_all(in_dim, hid_dim, lat_dim, edge_feat_dim, 
                     hid_edge_nn_dim, out_classes, hid_dim_classifier).to(device)
    model.apply(init_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay=1e-4, amsgrad=True)
    
    node_feat_transf = gen_node_features(feat_dim = in_dim)

    # checkpoints saving threshold on training loss - if have time implement this on acc or validation
    model_loss_min = float('inf')

    # TO BE IMPLEMENTED FOR LOGS AT LEAST 10
    logs_counter = 0
    
    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=node_feat_transf) #add_zeros
    test_loader = DataLoader(test_dataset, batch_size=bas, shuffle=False)

    # If train_path is provided then train on it 
    if args.train_path:
        print(f">> Starting the train of the model using the following train set: {args.train_path}")
        full_dataset = GraphDataset(args.train_path, transform=node_feat_transf)
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Now we can initialize the scheduler after dataset is created
        warmup_epochs = 3
        total_steps = (len(train_dataset) // bas) * num_epoches
        warmup_steps = (len(train_dataset) // bas) * warmup_epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        # Continue with the rest of the training code
        train_loader = DataLoader(train_dataset, batch_size=bas, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bas, shuffle=False)

        print(f"Training set size: {train_size}, Validation set size: {val_size}")

        # ----------- pre-training loop ------------ #
        print("\n--- Starting Pre-training of VGAE model ---")
        best_val_accuracy = 0.0
        for epoch in range(pretrain_epoches):
            train_loss = pretraining(model, train_loader, optimizer, device, epoch)
            train_accuracy, train_f1, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            val_accuracy, val_f1, _ = evaluate(val_loader, model, device, calculate_accuracy=True)
            
            if pretrain_epoches < 5 or (epoch + 1) % 5 == 0:
                print(f"PRETRAINING: Epoch {epoch + 1}/{pretrain_epoches}, Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            else:
                print(f"PRETRAINING: Epoch {epoch + 1}/{pretrain_epoches}, Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        # Training loop without KL parameters
        for epoch in range(num_epoches):
            train_loss = train(model, train_loader, optimizer, device, cur_epoch=epoch)
            train_accuracy, train_f1, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            val_accuracy, val_f1, _ = evaluate(val_loader, model, device, calculate_accuracy=True)
            
            if num_epoches < 5 or (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epoches}, Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epoches}, Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            # Save the checkpoint if validation accuracy improves
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                test_dir_name = os.path.basename(os.path.dirname(args.test_path))
                save_checkpoint(model, test_dir_name, epoch)
            else:
                patience_counter += 1
                
            # Early stopping based on F1 score
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best validation F1: {best_val_f1:.4f}")
                break
            
            # Step the scheduler after each batch
            scheduler.step()
    # Else if train_path NOT provided 
    if not args.train_path:
        checkpoint_path = args.checkpoint
        # raise an error if not able to find the checkpoint model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found! {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f">> Loading pre-training model from: {checkpoint_path}")
          
    # Evaluate and save test predictions
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    #output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_csv_path = os.path.join(KAGGLE_OUTPUT_PATH, f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

# arguments plus call to the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str,
                       default=os.path.join(KAGGLE_DATASET_PATH, "A", "train.json"))
    parser.add_argument("--test_path", type=str,
                       default=os.path.join(KAGGLE_DATASET_PATH, "A", "test.json"))
    args = parser.parse_args([])  # Empty list for Kaggle notebooks
    main(args)
