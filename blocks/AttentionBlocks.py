import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    
class ConvProjectionLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, channels_expansion=1, nb_heads=1, squeeze_factor=2, cls=False, last=False):
        super(ConvProjectionLayer, self).__init__()
        self.nb_heads = nb_heads
        self.out_channels = out_channels
        self.last = last
        self.cls_token = nn.Parameter(torch.randn(1, out_channels,1)) if cls else None
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm2d(in_channels))
        self.query_1D = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, groups=nb_heads)
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=squeeze_factor, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm2d(in_channels))
        self.key_1D = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, groups=nb_heads)
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=squeeze_factor, padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm2d(in_channels))
        self.value_1D = nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, groups=nb_heads)
        self.LN = nn.LayerNorm(out_channels)        
        self.FC_MLP = nn.Sequential(
            nn.Linear(out_channels, out_channels * channels_expansion),
            nn.GELU(),
            nn.Linear( out_channels * channels_expansion, out_channels))
        
        self.scale = 1/((out_channels // nb_heads)**0.5)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        #depthwise convolutions
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        #reshape the output to be used in the pointwise convolutions (linear layers because of the cls)
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        #add the cls token
        if self.cls_token is not None:
            q = torch.cat((self.cls_token, q), dim=-1)
            k = torch.cat((self.cls_token, k), dim=-1)
            v = torch.cat((self.cls_token, v), dim=-1)
        
        #pointwise convolutions
        q = self.query_1D(q)
        k = self.key_1D(k)
        v = self.value_1D(v)
        
        q = q.view(B, self.nb_heads, -1, q.size(-1))
        k = k.view(B, self.nb_heads, -1, k.size(-1))
        v = v.view(B, self.nb_heads, -1, v.size(-1))
        
        #attention mechanism
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = torch.einsum('bhij,bhdj->bhdi', attn, v)
        attn = attn.reshape(B, self.out_channels, -1)
        
        #add the cls token if needed
        x = x.reshape(B, C, -1)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token, x), dim=-1)

        x = x + attn
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.LN(x)
        
        x = self.FC_MLP(x)
        if self.cls_token is None: #normal case (no cls token)
            x = x.view(B, C, H, W)
            attn = attn.view(B, C, H, W)
            x= x+attn
        else: #cls token case
            x = x.permute(0, 2, 1)  # (B, C, T)
            if self.last: #last layer (cls token is returned as output)
                x = x[:,:,0]
                x = x+attn[:,:,0]
            else: #not the last layer (cls token is added to the input)
                x = x[:,:,1:]
                x = x.view(B, C, H, W)
                attn = attn[:,:,1:]
                attn = attn.view(B, C, H, W)
                x = x+attn
        return x