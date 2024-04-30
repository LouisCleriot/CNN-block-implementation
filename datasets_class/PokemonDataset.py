from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import torchvision

"""
Dataset from kaggle: https://www.kaggle.com/datasets/thedagger/pokemon-generation-one
Contains images of pokemons from generation one.
"""

class PokemonDataset(Dataset):
    def __init__(self, root_dir="datasets/pokemon2", transform=None, train=True, split_ratio=0.8):
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
        self.pokemon = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)
        
               
    def __len__(self):
        return len(self.pokemon)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, label = self.pokemon[idx]
        
        return img, label
    
    def get_classes(self):
        return self.pokemon.classes
    
    def get_nb_input_channels(self):
        return self.pokemon[0][0].shape[0]