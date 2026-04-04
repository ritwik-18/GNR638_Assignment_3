import torch
import cv2
import numpy as np
from model import UNet
import os

def calculate_dice(pred, truth):
    pred = (pred > 0.5).astype(np.float32)
    truth = (truth > 127).astype(np.float32)
    intersection = np.sum(pred * truth)
    return (2. * intersection + 1e-6) / (np.sum(pred) + np.sum(truth) + 1e-6)

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load('my_scratch_unet.pth', map_location=device))
    model.eval()

    total_dice = 0.0
    print("Evaluating model...")

    with torch.no_grad():
        for i in range(30):
            # Load images
            img = cv2.cvtColor(cv2.imread(f'data/imgs/{i}.png'), cv2.COLOR_BGR2RGB)
            img_t = torch.tensor(np.transpose(img / 255.0, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
            truth = cv2.imread(f'data/masks/{i}_mask.png', cv2.IMREAD_GRAYSCALE)
            
            # Predict
            output = torch.sigmoid(model(img_t)).squeeze().cpu().numpy()
            total_dice += calculate_dice(output, truth)

    print(f"Final Average Dice Score: {total_dice / 30:.4f}")

if __name__ == '__main__':
    evaluate()