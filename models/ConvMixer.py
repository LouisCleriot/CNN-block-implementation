import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.Patching import ConvMixerPatchEmbeding
from blocks.Archtectures import ConvMixerLayer

class ConvMixer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, kernel_size, depth, num_classes):
        super(ConvMixer, self).__init__()
        self.patch_embedding = ConvMixerPatchEmbeding(in_channels, embed_dim, patch_size)
        layers = nn.ModuleList()
        for _ in range(depth):
            layers.append(ConvMixerLayer(embed_dim,kernel_size))
        self.layers = nn.Sequential(*layers)
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.FC = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.layers(x)
        x = self.GlobalAvgPool(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        return x