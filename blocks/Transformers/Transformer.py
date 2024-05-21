import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.BasicBlocks import ResAdd
from blocks.Transformers.Patching import PatchEmbeddingConv
from blocks.AttentionBlocks import MultiHeadAttention
""" 
TODO : 

- Vit : Vision Transformer
- CvT : Convolutional Vision Transformer
- LeViT : Lightweight and Efficient Vision Transformer
- BoTNet : Bottleneck Transformers
- MobileViT : Mobile Vision Transformer
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
    
            
            
            