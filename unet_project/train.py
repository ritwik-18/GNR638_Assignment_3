import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import RealBiomedicalDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = RealBiomedicalDataset(imgs_dir='Pytorch-UNet/data/imgs', masks_dir='Pytorch-UNet/data/masks')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = UNet(n_channels=3, n_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    
    #Training
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

    torch.save(model.state_dict(), '/content/drive/MyDrive/my_scratch_unet.pth')
    print("Training complete! Saved safely to Google Drive.")

if __name__ == "__main__":
    train()