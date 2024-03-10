# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNBlock(nn.Module):
    """
    this block is an example of a simple conv-relu-conv-relu block
    with 3x3 convolutions
    """

    def __init__(self, in_channels):
        super(SimpleCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        return output


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an 
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """
    
    #Version ReLu-only pre-activation

    def __init__(self, in_channels,out_channels):
        super(ResidualBlock, self).__init__()
        #check if out_channels is even or odd
        midle_channels =  in_channels + out_channels 
        if midle_channels % 2 == 0:
            midle_channels = midle_channels // 2
        else:
            midle_channels = midle_channels // 2 + 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=midle_channels, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(midle_channels)
        self.conv2 = nn.Conv2d(in_channels=midle_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        output = F.relu(x)
        output = self.conv1(output)
        output = self.batchNorm1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.batchNorm2(output)
        
        output += self.skip(x)
        
        return output
        
        


class DenseBlock(nn.Module):
    """
    This block is the building block of the Dense network. It takes an
    input with in_channels, applies some blocks of convolutional, batchnorm layers
    and then concatenate the output with the original input
    """
    #Dense block with 3 layers (bn-relu-conv1-conv2)
    def __init__(self,in_channels):
        super(DenseBlock, self).__init__()
        self.batchNorm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
            
        output = self.batchNorm1(x)
        output = self.conv1(F.relu(output))
        
        output = self.batchNorm2(output)
        output = self.conv2(F.relu(output))
        
        output = self.batchNorm3(output)
        output = self.conv3(F.relu(output))
        
        output = torch.cat((x, output), 1)
           
        return output


class BottleneckBlock(nn.Module):
    """
    This block takes an input with in_channels reduces number of channels by a certain
    parameter "downsample" through kernels of size 1x1, 3x3, 1x1 respectively.
    """

    def __init__(self,in_channels,downsample):
        super(BottleneckBlock, self).__init__()
        
        median_channels = in_channels // downsample
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=median_channels, kernel_size=1, stride=1, padding=0)
        self.batchNorm1 = nn.BatchNorm2d(median_channels)
        self.conv2 = nn.Conv2d(in_channels=median_channels, out_channels=median_channels, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(median_channels)
        self.conv3 = nn.Conv2d(in_channels=median_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.batchNorm3 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        
        output = self.conv1(F.relu(x))
        output = self.batchNorm1(output)
        
        output = self.conv2(F.relu(output))
        output = self.batchNorm2(output)
        
        output = self.conv3(F.relu(output))
        output = self.batchNorm3(output)
        
        return output
    
class InceptionBlock(nn.Module):
    """
    This block takes an input with in_channels and applies 4 different
    convolutional layers to it. The output is the concatenation of the
    4 outputs.
    """

    def __init__(self,in_channels):
        super(in_channels, self).__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)

        out_1 = in_channels//4
        self.conv1x1_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_1, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=out_1, out_channels=3*out_1, kernel_size=3, stride=1, padding=1)
        
        self.conv1x1_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_1, kernel_size=1, stride=1, padding=0)
        self.conv5x5 = nn.Conv2d(in_channels=out_1, out_channels=(3*out_1)//2, kernel_size=1, stride=1, padding=0)
        
        self.conv1x1_4 = nn.Conv2d(in_channels=in_channels, out_channels=out_1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        
        branch_1 = F.relu(self.conv1x1_1(x))
        
        branch_2 = F.relu(self.conv3x3(F.relu(self.conv1x1_2(x))))
        
        branch_3 = F.relu(self.conv5x5(F.relu(self.conv1x1_3(x))))
        
        branch_4 = F.relu(self.conv1x1_4(F.max_pool2d(x, kernel_size=3, stride=1, padding=1)))
        
        output = torch.cat((branch_1, branch_2, branch_3, branch_4), 1)
        
        return output
    