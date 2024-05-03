import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiGridConv(nn.Module):
    """For each grid in the input pyramid, we rescale its neighboring grids to the same spatial resolution 
    and concatenate their feature channels. Convolution over the resulting single grid approximates 
    scale-space convolution on the original pyramid. Downscaling is max-pooling, while upscaling
    is nearest-neighbor interpolation. Ouput the same number different scale with the same
    number of channels for each. iT's advise to use a number of channels/2 for each scale 
    from the finest to the coarsest.
    https://arxiv.org/pdf/1611.07661
    
    Args:
        nb_scale (int): number of scales in the pyramid (number of input tensors)
        in_channels (tuple): number of input channels for each scale (from fine to coarse)
        out_channels (tuple): number of output channels for each scale (from fine to coarse)
    """
    
    def __init__(self, nb_scale, in_channels, out_channels=None):
        super(MultiGridConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
         
        self.branches_conv = nn.ModuleList()
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsampler = nn.MaxPool2d(2)
        
        self.branches_conv.append(nn.Sequential(
                nn.Conv2d(in_channels[0]+in_channels[1], out_channels[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels[0]),
                nn.ReLU(inplace=True)))
        for i in range(1,nb_scale-1):
            self.branches_conv.append(nn.Sequential(
                nn.Conv2d(in_channels[i-1]+in_channels[i]+in_channels[i+1], out_channels[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True)
            ))
        self.branches_conv.append(nn.Sequential(
            nn.Conv2d(in_channels[-2]+in_channels[-1], out_channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[-1]),
            nn.ReLU(inplace=True)))
        
    def forward(self, pyramid):
        x1 = pyramid[0]
        x2 = self.upsampler(pyramid[1])
        x = torch.cat((x1, x2), 1)
        output = [self.branches_conv[0](x)]
        for i in range(1,len(pyramid)-1):
            x1 = self.downsampler(pyramid[i-1])
            x2 = pyramid[i]
            x3 = self.upsampler(pyramid[i+1])
            out = torch.cat((x1, x2, x3), 1)
            output.append(self.branches_conv[i](out))
        x1 = self.downsampler(pyramid[-2])
        x2 = pyramid[-1]
        x = torch.cat((x1, x2), 1)
        output.append(self.branches_conv[-1](x))
        return output
            
class ASPPModule(nn.Module):
    """This module is used in deeplabv3 to capture multi-scale context by applying multiple parallel
    atrous convolution with different rates, a 1x1 conv and a global pooling. The output of the 
    parallel branches are concatenated and processed by a 1x1 convolution to reduce the number of channels.
    https://arxiv.org/pdf/1706.05587

    Args:
        in_channels (int): number of input channels
        size (tuple): input size (height, width)
        out_channels (int): number of output channels (256 in the original paper)
        rates (tuple): rates for the atrous convolution (6,12,18 in the original paper)
        branches_out_channels (tuple): number of output channels for each branch (64 each in the original paper)
    """
    def __init__(self, in_channels, size, out_channels=256, rates=(6,12,18), branches_out_channels=(64,64,64,64,64)):
        super(ASPPModule, self).__init__()
        h,w = size
        self.branches = nn.ModuleList()
        for rate in rates:
            rate_out_channel = branches_out_channels[rates.index(rate)]
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, rate_out_channel, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(rate_out_channel),
                nn.ReLU(inplace=True)))
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branches_out_channels[-1], kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(h,w), mode='bilinear')))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels,branches_out_channels[-2], kernel_size=1, bias=False),
            nn.BatchNorm2d(branches_out_channels[-2]),
            nn.ReLU(inplace=True)))
        self.conv = nn.Sequential(
            nn.Conv2d(sum(branches_out_channels), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        output = []
        for branch in self.branches:
            output.append(branch(x))
        output = torch.cat(output, 1)
        return self.conv(output)