import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """PatchEmbedding is a module that splits the input image into patches and
    then embeds the patches into linear vectors. The module can be used either
    with a fully connected layer or a convolutional layer. For the patching 
    part we use torch unfold function.
    Args:
        image_size (int or tuple): The size of the input image.
        patch_size (int or tuple): The size of the patch.
        in_channels (int): The number of channels in the input image.
        embed_dim (int): The dimension of the output embeddings.
        classifier_token (bool): Whether to add a classification token to the input embeddings.
    """
    def __init__(self,img_size, patch_size, in_channels, embed_dim, cls_token=True):
        super(PatchEmbedding, self).__init__()
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
            
        patchWidth, patchHeight = patch_size
        imgWidth, imgHeight = img_size
        numberOfPatches = (imgWidth*imgHeight)//(patchWidth*patchHeight)
        
        self.unfold = nn.Unfold(kernel_size=(patchWidth, patchHeight), stride=(patchWidth, patchHeight))
        self.linear = nn.Linear(patchWidth*patchHeight*in_channels, embed_dim)

        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
            numberOfPatches += 1
        else:
            self.cls_token = None
        
        self.pos_embedding = nn.Parameter(torch.randn(1,numberOfPatches,embed_dim))
        
    def forward(self, x):
        x = self.unfold(x)
        x = x.permute(0,2,1)
        x = self.linear(x)
        
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x += self.pos_embedding
        
        return x

class PatchEmbeddingConv(nn.Module):
    """PatchEmbeddingConv is a module create the patch embeddings using a convolutional
    layer. The module first applies a convolutional layer to the input image with a kernel
    and stride size equal to the patch size, and an output channel equal to the dimension of
    the embeddings. The output of the convolutional layer is then reshaped to create the patch
    embeddings. The module can also add a classification token to the input embeddings.

    Args:
        image_size (int or tuple): The size of the input image.
        patch_size (int or tuple): The size of the patch.
        in_channels (int): The number of channels in the input image.
        embed_dim (int): The dimension of the output embeddings.
        classifier_token (bool): Whether to add a classification token to the input embeddings. 
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, cls_token=True):
        super(PatchEmbeddingConv, self).__init__()
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
            
        patchWidth, patchHeight = patch_size
        imgWidth, imgHeight = img_size
        numberOfPatches = (imgWidth//patchWidth)*(imgHeight//patchHeight)
        
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
            numberOfPatches += 1
        else:
            self.cls_token = None
        
        self.pos_embedding = nn.Parameter(torch.randn(1,numberOfPatches,embed_dim))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x += self.pos_embedding
        
        return x
    
class ConvolutionalTokenEmbeding(nn.Module):
    """Module use in Convolutional Vision Transformer to process 
    a 2D image or a reshaped tensor into tokens map with new dimension
    (B,C,H,W) using convolutional layer. We then reshape the output tensor
    into a 1D tensor with shape (B, C, new_H * new_W) if the flag is set to true,
    which is not use in CvT and normalize the tensor using layer normalization.

    Args:
        input_channels (int): The number of channels in the input image.
        output_channels (int): The number of channels in the output tensor.
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolutional kernel.
        padding (int or tuple): The padding of the convolutional kernel.
        reshape (bool): Whether to reshape the output tensor or not.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, reshape=False):
        super(ConvolutionalTokenEmbeding, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(output_channels)
        self.reshape = reshape 
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        B, C, new_H, new_W = x.size()
        if self.reshape:
            x = x.view(B, new_H * new_W, C)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        return x