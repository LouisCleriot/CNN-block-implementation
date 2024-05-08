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
