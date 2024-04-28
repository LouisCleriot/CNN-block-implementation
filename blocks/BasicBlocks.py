import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an 
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """
    
    def __init__(self, in_channels,out_channels, depth_wise=True, bottleneck=False):
        super(ResidualBlock, self).__init__()
        midle_channels =  in_channels + out_channels 
        if midle_channels % 2 == 0:
            midle_channels = midle_channels // 2
        else:
            midle_channels = midle_channels // 2 + 1
            
        current = in_channels
        
        conv_layers = nn.ModuleList()
        
        if bottleneck:
            conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=midle_channels, kernel_size=1, stride=1, padding=0,bias=False))
            conv_layers.append(nn.BatchNorm2d(midle_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            current = midle_channels
        
        if depth_wise :
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=current, kernel_size=3, stride=1, padding=1, groups=current,bias=False))
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=out_channels, kernel_size=1, stride=1, padding=0,bias=False))
        else:
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=False))
        
        conv_layers.append(nn.BatchNorm2d(out_channels))    
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        self.mainbranch = nn.Sequential(*conv_layers)
        self.skip = nn.Sequential()
        if in_channels != out_channels :
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,bias=False)
        
    def forward(self, x):
        output = self.mainbranch(x)
        output += self.skip(x)
        
        return F.relu(output)
        
        


class DenseBlock(nn.Module):
    """
    This block is the building block of the Dense network. It takes an
    input with in_channels, applies some blocks of convolutional, batchnorm layers
    and then concatenate the output with the original input
    """
    #Dense block with 3 layers (bn-relu-conv1-conv2)
    def __init__(self, in_channels, depth_wise=True, bottleneck=False):
        super(DenseBlock, self).__init__()
        midle_channels =  in_channels 
        if in_channels % 2 == 0:
            midle_channels = midle_channels // 2
        else:
            midle_channels = midle_channels // 2 + 1
            
        current = in_channels
        
        conv_layers = nn.ModuleList()
        
        if bottleneck:
            conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=midle_channels, kernel_size=1, stride=1, padding=0,bias=False))
            conv_layers.append(nn.BatchNorm2d(midle_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            current = midle_channels
        
        if depth_wise :
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=current, kernel_size=3, stride=1, padding=1, groups=current,bias=False))
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=in_channels, kernel_size=1, stride=1, padding=0,bias=False))
        else:
            conv_layers.append(nn.Conv2d(in_channels=current, out_channels=in_channels, kernel_size=3, stride=1, padding=1,bias=False))
        
        conv_layers.append(nn.BatchNorm2d(in_channels))    
        conv_layers.append(nn.ReLU(inplace=True))
        self.dense = nn.Sequential(*conv_layers)
        self.regularise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(nn.BatchNorm2d(in_channels)*2),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
            
        output = self.dense(x)
        output = torch.cat((x, output), 1)
        output = self.regularise(output)
           
        return output


class ConvBottleneck(nn.Module):
    """
    This block takes an input with in_channels reduces number of channels by a certain
    parameter "downsample" through kernels of size 1x1, 3x3, 1x1 respectively.
    """

    def __init__(self,in_channels,downsample, depth_wise=False, residual=False):
        super(ConvBottleneck, self).__init__()
        conv_layers = nn.ModuleList()
        median_channels = in_channels // downsample
        self.residual = residual
        conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=median_channels, kernel_size=1, stride=1, padding=0,bias=False))
        conv_layers.append(nn.BatchNorm2d(median_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        if depth_wise :
            conv_layers.append(nn.Conv2d(in_channels=median_channels, out_channels=median_channels, kernel_size=3, stride=1, padding=1, groups=median_channels,bias=False))
            conv_layers.append(nn.Conv2d(in_channels=median_channels, out_channels=median_channels, kernel_size=1, stride=1, padding=0,bias=False))
        else :  
            conv_layers.append(in_channels=median_channels, out_channels=median_channels, kernel_size=3, stride=1, padding=1)
        conv_layers.append(nn.BatchNorm2d(median_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.Conv2d(in_channels=median_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        if not residual:
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv_blocks = nn.Sequential(*conv_layers)        
    def forward(self, x):
        
        output = self.conv_blocks(x)
        if self.residual:
            output += x
            output = F.relu(output)
        return output