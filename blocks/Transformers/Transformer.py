import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.BasicBlocks import ResAdd
from blocks.Transformers.Patching import PatchEmbeddingConv
from blocks.AttentionBlocks import MultiHeadAttention
""" 
TODO : 

- CvT : Convolutional Vision Transformer
- LeViT : Lightweight and Efficient Vision Transformer
- BoTNet : Bottleneck Transformers
- MobileViT : Mobile Vision Transformer
- ViLT
"""
        
        
        
class VisionTransformer(nn.Module):
    """Vision Transformer is the adaptation of the Transformer model to the
    computer vision domain. It begin by splitting the input image into patches,
    using a feedforward neural network to embed the patches into linear vectors,
    and then feeding these vectors into a standard Transformer model. To this vector
    we concatenate a learnable positional encoding to the input embeddings and add 
    a classification token to the input embeddings. The classification token is used
    to aggregate information from the output of the Transformer model and make a prediction.
    At the output of the Transformer model, we apply a linear layer to the output of the
    classification token to make a prediction. 
    Args:
        image_size (int or tuple): The size of the input image.
        patch_size (int or tuple): The size of the patch.
        num_classes (int): The number of classes in the dataset.
        dim (int): The dimension of the input embeddings.        
    """
    def __init__(self, image_size, patch_size, in_channels, embed_dim, L, heads, n_class, cls_token=True, dk=None, dv=None):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbeddingConv(image_size, patch_size, in_channels, embed_dim, cls_token)
        
        encoder = nn.ModuleList()
        for _ in range(L):
            module1 = nn.Sequential(
                nn.LayerNorm(embed_dim),
                MultiHeadAttention(dmodel=embed_dim,heads=heads, dk=dk, dv=dv)
                )
            encoder.append(nn.Sequential(ResAdd(module1)))
            module2 = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU()
                )
            encoder.append(nn.Sequential(ResAdd(module2)))
        self.encoder = nn.Sequential(*encoder)
        self.MLPHead = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, (embed_dim+n_class)//2),
            nn.ReLU(),
            nn.Linear((embed_dim+n_class)//2, n_class)
            )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = x[:,0]
        x = self.MLPHead(x)
        return x

from blocks.Transformers.Patching import ConvolutionalTokenEmbeding
from blocks.AttentionBlocks import ConvProjectionLayer
            
class ConvolutionalVisionTransformerStage(nn.Module):
    """Convolutional Vision Transformer is the adaptation of the Vision Transformer model using 
    convolutional layers to tokenise patches of the input image. The ConvProjectionLayer
    will use this tokenised patches to compute the attention weights using spatial information
    and the MultiHeadAttention mechanism. Like other Transformer models, we use a classification
    token to aggregate information from the output of the Transformer model and make a prediction.
    at the end of the model with a MLPHead.
    https://arxiv.org/pdf/2103.15808
    
    Args:
        patch_in_channels (int): The number of channels in the input patches.
        patch_out_channels (int): The number of channels in the output patches.
        patch_kernel_size (int): The size of the kernel in the convolutional layer.
        patch_stride (int): The stride of the convolutional layer.
        MH_heads (int): The number of heads in the MultiHeadAttention mechanism.
        expand_channels (int): The factor by which to expand the number of channels in the MLP of the convprojection.
        n_ConvProjection (int): The number of ConvProjectionLayer in this stage.
        squeeze_factor (int): The factor by which to reduce the number of channels in the convolutional layer.
        cls_token (bool): Whether to add a classification token to the input embeddings.
        nb_classes (int): The number of classes in the dataset. 
    """
    
    def __init__(self, patch_in_channels, patch_out_channels, patch_kernel_size, patch_stride, MH_heads, expand_channels, n_ConvProjection, squeeze_factor=2, cls_token=False, nb_classes=1000):
        super(ConvolutionalVisionTransformerStage, self).__init__()
        
        padding = patch_kernel_size // 2
        self.patch_embedding = ConvolutionalTokenEmbeding(patch_in_channels, patch_out_channels, patch_kernel_size, patch_stride, padding)
        conv_projection = nn.ModuleList()
        last = False
        for i in range(n_ConvProjection):
            if cls_token:
                last = i == n_ConvProjection - 1
            conv_projection.append(
                ConvProjectionLayer(patch_out_channels, patch_out_channels, 3, expand_channels, MH_heads, squeeze_factor, cls_token, last)
                )
        if cls_token:
            mlp = nn.Sequential(
                nn.LayerNorm(patch_out_channels),
                nn.Linear(patch_out_channels, patch_out_channels * expand_channels),
                nn.GELU(),
                nn.Linear(patch_out_channels * expand_channels, nb_classes)
                )
            conv_projection.append(mlp)
        self.conv_projection = nn.Sequential(*conv_projection)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.conv_projection(x)
        return x
            