import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.Patching import ConvMixerPatchEmbeding

class ConvMixerLayer(nn.Module):
    """
    ConvMixerLayerLayer is a module that applies a depthwise separable convolution
    followed by a GELU activation function and a pointwise convolution to the input.
    Args:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The number of channels in the output tensor.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolutional kernel.
    """
    def __init__(self, embed_dim, kernel_size=9):
        super(ConvMixerLayer, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding=kernel_size//2, groups=embed_dim),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim))
            
        self.pointwise = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim))
                
    def forward(self, x):
        x = self.depthwise(x) + x
        x = self.pointwise(x)
        return x