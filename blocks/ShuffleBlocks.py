import torch
import torch.nn as nn
import torch.nn.functional as F

class InterleavedModule(nn.Module):
    """
    Module use to intertwine multiple tensor. It will put the first channel
    pf the first group, then the first of the seconde, then the first of the 
    third, .... 
    """
    def __init__(self):
        super(InterleavedModule,self).__init__()            
        
    def forward(self,*args):
        #kwargs is used for case were we want to interleave more than 2 inputs
        
        total = len(args)
     
        batch_size, channels, height, width = args[0].size()
        
        interleaved = torch.empty(batch_size, channels*total, height, width, device=args[0].device)
        for i, tensor in enumerate(args):
            interleaved[:, i::total, :, :] = tensor

        return interleaved
    
class ShuffleModule(nn.Module):
    """
    this block take 2 parameter : groups number and in_channels. first 
    there is a 1x1 group convolution, then a channel shuffle operation, a depthwise
    3x3 convolution and finally a 1x1 group convolution. 
    https://arxiv.org/pdf/1707.01083	
    """
    def __init__(self,groups,in_channels,stride=1, dense=False, residual=False):
        super(ShuffleModule,self).__init__()
        self.groups = groups
        self.dense = dense
        self.residual = residual
        self.interleaved = InterleavedModule()
        pooling = False
        if stride != 1 :
            pooling = True
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, groups=groups,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, groups=groups,bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.pool = nn.Sequential()
        if pooling:
            self.pool = nn.AdaptiveAvgPool2d(3, stride=stride, padding=1)
    def forward(self, x):
        output = self.conv1x1(x)
        list_groups = torch.chunk(output, self.groups, 1)
        output = self.interleaved(*list_groups)
        output = self.block(output)
        outputParallel = self.pool(x)
        if self.residual : 
            output += outputParallel
        if self.dense :
            output = torch.cat((x, output), 1)
        return output
    
class InterleavedGroupConvolutionModule(nn.Module):
    """Interleaved group convolutions consist of two group con-
    volutions, primary group convolution and secondary group
    convolution. Primary group convolutions to handle spatial correlation,
    secondary group convolution to blend the channels. 
    L primary partitions of M channels each, 
    M secondary partitions of L channels each.
    https://arxiv.org/pdf/1707.02725.pdf
    Args:
        L (int): the number of primary partitions 
        M (int): the number of secondary partitions
        in_channels (int): the number of input channels
    """
    def __init__(self, L, M, in_channels):
        super(InterleavedGroupConvolutionModule, self).__init__()
        self.L = L
        self.M = M
        self.in_channels = in_channels
        self.primary = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=L,bias=False, padding=1)
        self.secondary = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=M,bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.interleaved = InterleavedModule()
        
    def forward(self, x):
        output = self.primary(x)
        # split the output into L groups
        list_groups = torch.chunk(output, self.L, 1)
        # interleave the groups
        output = self.interleaved(*list_groups)
        output = self.secondary(output)
        # split the output into M groups
        list_groups = torch.chunk(output, self.M, 1)
        # interleave the groups
        output = self.interleaved(*list_groups)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
