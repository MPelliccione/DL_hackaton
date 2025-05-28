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



def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
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
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # create directories
    create_dirs()
    
    # Hyperparameters for the model (circa a ctrl+c - ctrl+v from competiton GitHub)
    in_dim  = 32           # previous val: 128 i want a faster model
    hid_dim = 128
    lat_dim = 16            # 16
    out_classes = 6
    edge_feat_dim=7
    hid_edge_nn_dim=32
    hid_dim_classifier=64
    
    pretrain_epoches = 20
    num_epoches = 20
    learning_rate = 0.0005 # Reduced from 0.0005
    bas = 32            # Reduced from 32
    
    # Remove unused KL parameters
    torch.manual_seed(0)
    
    # Initialize model with weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    model = VGAE_all(in_dim, hid_dim, lat_dim, edge_feat_dim, 
                     hid_edge_nn_dim, out_classes, hid_dim_classifier).to(device)
    model.apply(init_weights)

    # Use AdamW instead of Adam
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
        train_dataset = GraphDataset(args.train_path, transform=node_feat_transf) #add_zeros
        train_loader = DataLoader(train_dataset, batch_size=bas, shuffle=True)

        # ----------- pre-training loop ------------ #
        print("\n--- Starting Pre-training of VGAE model ---")
        for epoch in range(pretrain_epoches):
            train_loss = pretraining(model, train_loader, optimizer, device, epoch)
            train_accuracy, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"PRETRAINING: Epoch {epoch + 1}/{pretrain_epoches}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # Training loop without KL parameters
        for epoch in range(num_epoches):
            train_loss = train(model, train_loader, optimizer, device, cur_epoch=epoch)
            train_accuracy, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epoches}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

            # Save the checkpoint if condition
            if (epoch < 5) or (train_loss < model_loss_min):
                model_loss_min = train_loss
                test_dir_name = os.path.basename(os.path.dirname(args.test_path))
                save_checkpoint(model, test_dir_name, epoch)

        # SAVE LOGS EACH 10 EPOCHS TO BE COMPLETED 
        #logs/: Log files for each training dataset. Include logs of accuracy and loss recorded every 10 epochs. # usare sempre test_dir_name
        
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
    parser.add_argument("--train_path", type=str
                      )
    parser.add_argument("--test_path", type=str
                      )
    args = parser.parse_args([])  # Empty list for Kaggle notebooks
    main(args)
