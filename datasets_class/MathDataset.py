from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import torchvision

"""
Dataset from kaggle: https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
Contains images of number from 0 to 9 as well as the four basic operations, equal sign, 
not equal sign and variable x, y and z.
"""

class MathDataset(Dataset):
    def __init__(self, root_dir="datasets/math", transform=None, train=True, split_ratio=0.8):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.root_dir = os.path.join(root_dir, "Train")
        else:
            self.root_dir = os.path.join(root_dir, "Test")
        self.root_dir = root_dir
        self.transform = transform
        self.math = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)
        
               
    def __len__(self):
        return len(self.math)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, label = self.math[idx]
        
        return img, label
    
    def get_classes(self):
        return self.math.classes
    
    def get_nb_input_channels(self):
        return self.math[0][0].shape[0]