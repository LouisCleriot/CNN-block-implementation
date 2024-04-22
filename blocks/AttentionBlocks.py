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
    
    def __init__(self,in_channels,downsample=2):
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
    
class EfficientChannelAttention(nn.Module):
    """
    same principle as the squeeze and excitation block but with a more efficient
    implementation. It takes an input with in_channels, transform it into a 1d tensor
    and apply 1 conv1d layer with a sigmoid activation, the result is the multiplied by the input.   	
    https://doi.org/10.48550/arXiv.1910.03151
    """
    
    def __init__(self,in_channels):
        super(EfficientChannelAttention,self).__init__()
        self.attentionlayer = nn.Conv1d(1,1,kernel_size=5,stride=1,padding=2)
        
    def forward(self,x):
        #global average pooling
        output = F.adaptive_avg_pool2d(x, (1,1))
        #transform into 1d tensor
        output = output.view(output.size(0), -1)
        output = output.unsqueeze(1)
        #go through the conv1d layer 
        output = self.attentionlayer(output)
        #apply sigmoid to get the attention weights
        output = F.sigmoid(output)
        #modulate the input with the attention weights
        output = output.view(output.size(0), -1, 1, 1)
        output = x * output
        return output