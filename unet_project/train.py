import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import ToyDataset

def train():
    # 1. Setup device, dataset, and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = ToyDataset(num_samples=200, img_size=128) # Smaller size for faster local testing
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Initialize Model, Loss, and Optimizer
    # n_channels=1 (grayscale input), n_classes=1 (binary mask)
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    # BCEWithLogitsLoss is best for binary segmentation (it applies sigmoid internally)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()