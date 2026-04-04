import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class RealBiomedicalDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [str(i) for i in range(30)] 

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        img_file = os.path.join(self.imgs_dir, f'{idx}.png')
        mask_file = os.path.join(self.masks_dir, f'{idx}_mask.png')
        
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        img = np.transpose(img / 255.0, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask / 255.0, axis=0).astype(np.float32)
        
        return torch.tensor(img), torch.tensor(mask)