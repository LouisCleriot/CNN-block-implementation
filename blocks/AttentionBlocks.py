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
        return output
    
class EfficientChannelAttention(nn.Module):
    """
    same principle as the squeeze and excitation block but with a more efficient
    implementation. It takes an input with in_channels, transform it into a 1d tensor
    and apply 1 conv1d layer with a sigmoid activation, the result is the multiplied by the input.   	
    https://doi.org/10.48550/arXiv.1910.03151
    """
    
    def __init__(self,in_channels,gamma=2,beta=1):
        super(EfficientChannelAttention,self).__init__()
        t = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + beta) / gamma))
        kernel = t if t % 2 != 0 else t + 1
        self.attentionlayer = nn.Conv1d(1,1,kernel_size=t,stride=1,padding=kernel//2, bias=False)

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
    
""" 
TODO: - CBAM : Convolutional Block Attention Module
      - STN : Spatial Transformer Network
      - SA : Self Attention / Intra-Attention
"""
class SelfAttention(nn.Module):
    """Attention mechanism used in the Transformer model. It's used to compute the
    attention weights between elements in the input sequence. We use 3 matrices to
    compute the attention weights: the query matrix, the key matrix and the value matrix.
    Args:
        dk (int): The dimension of the key and query matrix.
        dv (int): The dimension of the value matrix.
        dmodel (int): The dimension of the input embeddings.
        heads (int): The number of heads in the multi-head attention.
        parallel (bool): Whether to do the computation in parallel or not.
    """
    def __init__(self,dmodel, dk, dv):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(dmodel,dk,bias=False)
        self.key = nn.Linear(dmodel,dk,bias=False)
        self.value = nn.Linear(dmodel,dv,bias=False)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention = torch.matmul(query,key.transpose(-2,-1))
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention,value)
        
        return output
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism used in the Transformer model. It's used to compute the
    attention weights between elements in the input sequence. We use 3 matrices to
    compute the attention weights: the query matrix, the key matrix and the value matrix.
    Args:
        dk (int): The dimension of the key and query matrix.
        dv (int): The dimension of the value matrix.
        dmodel (int): The dimension of the input embeddings.
        heads (int): The number of heads in the multi-head attention.
    """
    def __init__(self,dmodel,heads, dk=None, dv=None):
        super(MultiHeadAttention, self).__init__()
        dk = dk if dk else dmodel // heads
        dv = dv if dv else dmodel // heads
        
        self.heads =nn.ModuleList([SelfAttention(dmodel, dk, dv) for _ in range(heads)])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads],dim=-1)
    