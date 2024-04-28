import torch
import torch.nn as nn
import torch.nn.functional as F

#import attention blocks
from blocks.AttentionBlocks import SqueezeAndExciteBlock, EfficientChannelAttention
class FireBlock(nn.Module):
    """ 
    takes an input with in_channels and applies a squeeze layer 
    to reduce the number of channels to s1, then applies an expand layer
    with e1 and e3 number of channels, respectively.
    e1 is the number of 1x1 filters and e3 is the number of 3x3 filters.
    we have s1<=e1+e3
    https://doi.org/10.48550/arXiv.1602.07360
    """
    def __init__(self,in_channels,squeeze,expand1,expand3):
        super(FireBlock,self).__init__()
        
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=squeeze, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True)
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze, out_channels=expand1, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True)
        )
        self.expand2 = nn.Sequential( 
            nn.Conv2d(in_channels=squeeze, out_channels=squeeze, kernel_size=3, stride=1, padding=1, groups=squeeze,bias=False),
            nn.Conv2d(in_channels=squeeze, out_channels=expand3, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(expand3),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self,x):
        squeeze = self.squeeze(x)
        expand1 = self.expand1(squeeze)
        expand2 = self.expand2(squeeze)
        output = torch.cat((expand1,expand2),1)
        return output
    
class SlimConv(nn.Module):
    """Reduce and reform feature channels to improve the quality of feature representations.
    The SlimConv reduces calculations and maintains the capability of feature representations.
    It's composed of an attention mechanism, 2 branches of convolutions, and a concatenation.
    the first branch is a 3x3 convolution, the second is a 1x1 followed by a 3x3 convolution.
    https://ieeexplore.ieee.org/document/9477103
    
    Args:
        attentionMecanism (string): can be 'SE', 'ECA', default is SE
        in_channels (int): the number of input channels
        (Kup,Klow) (tuple of int): the reduction ratio for each branch, the default is (2,4)
        downsample : use if SE corresponds to the reduction of the 2nd fc layer
        gamma : use if ECA corresponds to the gamma value (mapping function)
        beta : use if ECA corresponds to the beta value (mapping function)
    """
    def __init__(self, in_channels, attentionMecanism='SE', k=(2,4), downsample=2, gamma=2, beta=1):
        super(SlimConv, self).__init__()
        kup = k[0]
        klow = k[1]
        
        if attentionMecanism == 'SE':
            median_channels = in_channels // downsample
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, median_channels),
                nn.ReLU(inplace=True),
                nn.Linear(median_channels, in_channels),
                nn.Sigmoid(),
                nn.Unflatten(1, (in_channels, 1, 1))
            )
        else  :
            t = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + beta) / gamma))
            kernel = t if t % 2 != 0 else t + 1
            self.attention = nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Conv1d(1,1,kernel_size=kernel,stride=1,padding=kernel//2, bias=False),
                nn.Sigmoid(),
                nn.Unflatten(1, (in_channels, 1, 1))
            )
        self.upperBranch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//kup, kernel_size=3, stride=1, padding=1,bias=False),
        )
        self.lowerBranch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//klow, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(in_channels//klow),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels//klow, out_channels=in_channels//klow, kernel_size=3, stride=1, padding=1,bias=False),
        )
        self.BN = nn.BatchNorm2d(in_channels//kup + in_channels//klow) 
    def forward(self, x):
        #calculate the attention weights
        attention = self.attention(x)
        attentionFlip = torch.flip(attention, [1])
        #modulate the input with the attention weights
        xupper = x * attention
        xlower = x * attentionFlip
        #split in half and add each x
        xupper1,xupper2 = torch.chunk(xupper,2,1)
        xupper = xupper1 + xupper2 
        xlower1,xlower2 = torch.chunk(xlower,2,1)
        xlower = xlower1+xlower2
        #go through each branch
        upper = self.upperBranch(xupper)
        lower = self.lowerBranch(xlower)
        output = F.relu(self.BN(torch.cat((upper, lower), 1)))
        return output