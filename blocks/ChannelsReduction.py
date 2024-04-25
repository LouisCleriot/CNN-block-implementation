import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(in_channels=in_channels, out_channels=squeeze, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True)
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze, out_channels=expand1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True)
        )
        self.expand2 = nn.Sequential( 
            nn.Conv2d(in_channels=squeeze, out_channels=squeeze, kernel_size=3, stride=1, padding=1, groups=squeeze),
            nn.Conv2d(in_channels=squeeze, out_channels=expand3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expand3),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self,x):
        squeeze = self.squeeze(x)
        expand1 = self.expand1(squeeze)
        expand2 = self.expand2(squeeze)
        output = torch.cat((expand1,expand2),1)
        return output