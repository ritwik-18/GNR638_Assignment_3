import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class ToyDataset(Dataset):
    def __init__(self, num_samples=100, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a black image
        image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Randomize circle properties
        center_x = np.random.randint(40, self.img_size - 40)
        center_y = np.random.randint(40, self.img_size - 40)
        radius = np.random.randint(10, 40)
        
        # Draw a white circle
        cv2.circle(image, (center_x, center_y), radius, 1.0, -1)
        
        # For this toy task, the mask is exactly the same as the image
        mask = image.copy()
        
        # Add a channel dimension for PyTorch (C, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.tensor(image), torch.tensor(mask)