import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExciteBlock(nn.Module):
    """
    This block takes an input with in_channels and applies a squeeze and excitation
    operation to it. It first applies a global average pooling to the input, then
    applies 2 fully connected layers with ReLU activations and finally applies a
    sigmoid activation to the output. It multiplies the input by the output.
    """
    
    def __init__(self,in_channels,downsample):
        super(SqueezeAndExciteBlock,self).__init__()
        median_channels = in_channels // downsample
        self.fully_connected_1 = nn.Linear(in_channels, median_channels)
        self.fully_connected_2 = nn.Linear(median_channels, in_channels)
        
    def forward(self, x):
        
        output = F.adaptive_avg_pool2d(x, (1,1))
        output = output.view(output.size(0), -1)
        output = F.relu(self.fully_connected_1(output))
        output = self.fully_connected_2(output)
        output = F.sigmoid(output)
        
        output = output.view(output.size(0), -1, 1, 1)
        output = x * output
        output = torch.cat((x, output), 1)
        return output