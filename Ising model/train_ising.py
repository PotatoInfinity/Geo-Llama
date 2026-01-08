import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from geo_transformer.core import conformal_lift, inner_cl41
from geo_transformer.model import GeometricTransformer
import os

# Hyperparameters
EMBED_DIM = 8
N_HEADS = 2
N_LAYERS = 1
N_CLASSES = 3
BATCH_SIZE = 8 
EPOCHS = 5
LR = 0.001

def train():
    # Load data
    data_path = "/Users/mac/Desktop/Geo-llama/Research/data/ising_data.pt"
    if not os.path.exists(data_path):
        print("Data not found. Run Research/ising/data_gen.py first.")
        return
        
    checkpoint = torch.load(data_path)
    X_raw = checkpoint['data'] # (N, 8, 8)
    Y = checkpoint['labels']
    
    X_flat = X_raw.view(X_raw.shape[0], -1) 
    
    print("Lifting spins to Conformal Null Basis...")
    X_lifted = conformal_lift(X_flat) 
    X_input = X_lifted.unsqueeze(2).repeat(1, 1, EMBED_DIM, 1) # (N, 64, 8, 32)
    
    dataset = TensorDataset(X_input, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model with RecursiveRotorAccumulator disabled for debugging
    model = GeometricTransformer(EMBED_DIM, N_HEADS, N_LAYERS, N_CLASSES, use_rotor_pool=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_rotor_norm = 0
        correct = 0
        class_correct = [0] * N_CLASSES
        class_total = [0] * N_CLASSES
        
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                norm_sq = inner_cl41(x_batch, x_batch).mean()
                total_rotor_norm += torch.sqrt(torch.abs(norm_sq)).item()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            
            for i in range(len(y_batch)):
                label = y_batch[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1
            
        acc = correct / len(dataset)
        avg_loss = total_loss / len(loader)
        avg_rotor_norm = total_rotor_norm / len(loader)
        
        class_accs = [class_correct[i]/class_total[i] if class_total[i] > 0 else 0 for i in range(N_CLASSES)]
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | R-Norm: {avg_rotor_norm:.6f}")
        print(f"  Class Accs: Ordered: {class_accs[0]:.3f}, Critical: {class_accs[1]:.3f}, Disordered: {class_accs[2]:.3f}")


    print("Training Complete.")

if __name__ == "__main__":
    train()
