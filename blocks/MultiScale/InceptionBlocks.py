import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InceptionModuleV1(nn.Module):
    """
    This block takes an input with in_channels and applies 4 different
    convolutional layers to it. The output is the concatenation of the
    4 outputs.
    """

    def __init__(self,in_channels, out1x1, out3x3, out5x5, outmaxpool, downsample3x3=2,downsample5x5=8):

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1, stride=1, padding=0,bias=False)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=np.ciel(in_channels//downsample3x3), kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(np.ciel(in_channels//downsample3x3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=np.ciel(in_channels//downsample3x3), out_channels=out3x3, kernel_size=3, stride=1, padding=1,bias=False),
            )
        
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=np.ciel(in_channels//downsample5x5), kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(np.ciel(in_channels//downsample5x5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=np.ciel(in_channels//downsample5x5), out_channels=out5x5, kernel_size=5, stride=1, padding=2,bias=False),
            )
        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=outmaxpool, kernel_size=1, stride=1, padding=0,bias=False),
            )
        
        self.bn = nn.BatchNorm2d(out1x1+out3x3+out5x5+outmaxpool)
        
    def forward(self, x):
            
        branch_1 = self.conv1x1(x)
        branch_2 = self.conv3x3(x)
        branch_3 = self.conv5x5(x)
        branch_4 = self.maxpool(x)
        output = F.relu(self.bn(torch.cat((branch_1, branch_2, branch_3, branch_4), 1)))
        
        return output
    
class InceptionModuleV2Base(nn.Module):
    """ 
    Same module as the InceptionModuleV1 but use 2 3x3 instead of 1 5x5
    convolution. 
    """
    def __init__(self,in_channels, out1x1, out3x3, out2_3x3, outmaxpool, downsample3x3=2,downsample2_3x3=4):
        
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1, stride=1, padding=0,bias=False)
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=np.ciel(in_channels//downsample3x3), kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(np.ciel(in_channels//downsample3x3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=np.ciel(in_channels//downsample3x3), out_channels=out3x3, kernel_size=3, stride=1, padding=1,bias=False)
            )
        
        self.conv2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=np.ciel(in_channels//downsample2_3x3), kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(np.ciel(in_channels//downsample2_3x3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=np.ciel(in_channels//downsample2_3x3), out_channels=out2_3x3, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out2_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out2_3x3, out_channels=out2_3x3, kernel_size=3, stride=1, padding=1,bias=False)
            )
        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=outmaxpool, kernel_size=1, stride=1, padding=0,bias=False)
            )
        
        self.bn = nn.BatchNorm2d(out1x1+out3x3+out2_3x3+outmaxpool)
        
    def forward(self, x):
        branch_1 = self.conv1x1(x)
        branch_2 = self.conv3x3(x)
        branch_3 = self.conv2_3x3(x)
        branch_4 = self.maxpool(x)
        output = F.relu(self.bn(torch.cat((branch_1, branch_2, branch_3, branch_4), 1)))
        return output
    
class InceptionModuleV2Factorize(nn.Module):
    """
    module that factorise all nxn convolution into 1xn and nx1 convolution
    with n a parameter of the module. 
    """
    def __init__(self,in_channels,n, out1x1, outnxn, out2_nxn, outmaxpool):
        pad = n//2
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1, stride=1, padding=0,bias=False)
        self.convnxn = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=outnxn, kernel_size=1, stride=1, padding=0,bias=False),
                            nn.BatchNorm2d(outnxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=outnxn, out_channels=outnxn, kernel_size=(1,n), stride=1, padding=pad,bias=False),
                            nn.BatchNorm2d(outnxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=outnxn, out_channels=outnxn, kernel_size=(n,1), stride=1, padding=pad,bias=False))
        self.conv2_nxn = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out2_nxn, kernel_size=1, stride=1, padding=0,bias=False),
                            nn.BatchNorm2d(out2_nxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=out2_nxn, out_channels=out2_nxn, kernel_size=(1,n), stride=1, padding=pad,bias=False),
                            nn.BatchNorm2d(out2_nxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=out2_nxn, out_channels=out2_nxn, kernel_size=(n,1), stride=1, padding=pad,bias=False),
                            nn.BatchNorm2d(out2_nxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=out2_nxn, out_channels=out2_nxn, kernel_size=(1,n), stride=1, padding=pad,bias=False),
                            nn.BatchNorm2d(out2_nxn),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=out2_nxn, out_channels=out2_nxn, kernel_size=(n,1), stride=1, padding=pad,bias=False))
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=outmaxpool, kernel_size=1, stride=1, padding=0,bias=False)
            )
        self.bn = nn.BatchNorm2d(out1x1+outnxn+out2_nxn+outmaxpool)
    def forward(self, x):
        branch_1 = self.conv1x1(x)
        branch_2 = self.convnxn(x)
        branch_3 = self.conv2_nxn(x)
        branch_4 = self.maxpool(x)
        output = F.relu(self.bn(torch.cat((branch_1, branch_2, branch_3, branch_4), 1)))
        return output

class InceptionModuleV2Wide(nn.Module):
    """
    Widest module of the InceptionV2 to avoid representational bottleneck
    """
    def __init__(self,in_channels, out1x1, out3x3, out2_3x3, outmaxpool):
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=1, stride=1, padding=0,bias=False)
        
        self.branch2_3x3Part1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out3x3, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out3x3, out_channels=out3x3, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True))
        self.branch2_3x3Part2A = nn.Conv2d(in_channels=out3x3, out_channels=out3x3, kernel_size=(1,3), stride=1, padding=(0,1),bias=False)
        self.branch2_3x3Part2B = nn.Conv2d(in_channels=out3x3, out_channels=out3x3, kernel_size=(3,1), stride=1, padding=(1,0),bias=False)
        
        self.branch3x3Part1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out2_3x3, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out2_3x3),
            nn.ReLU(inplace=True))
        self.branch3x3Part2A = nn.Conv2d(in_channels=out2_3x3, out_channels=out2_3x3, kernel_size=(1,3), stride=1, padding=(0,1),bias=False)
        self.branch3x3Part2B = nn.Conv2d(in_channels=out2_3x3, out_channels=out2_3x3, kernel_size=(3,1), stride=1, padding=(1,0),bias=False)
        
        self.maxpool = nn.Sequential( 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=outmaxpool, kernel_size=1, stride=1, padding=0,bias=False)
            )
        
        self.bn = nn.BatchNorm2d(out1x1+out3x3+out2_3x3+outmaxpool)
    
    def forward(self, x):
        branch_1 = self.conv1x1(x)
        branch_2 = self.branch2_3x3Part1(x)
        branch_2 = F.relu(torch.cat((self.branch2_3x3Part2A(branch_2), self.branch2_3x3Part2B(branch_2)), 1))
        branch_3 = self.branch3x3Part1(x)
        branch_3 = F.relu(torch.cat((self.branch3x3Part2A(branch_3), self.branch3x3Part2B(branch_3)), 1))
        branch_4 = self.maxpool(x)
        output = F.relu(self.bn(torch.cat((branch_1, branch_2, branch_3, branch_4), 1)))
        return output
    
class InceptionModulev2Pooling(nn.Module):
    """
    Inception module that reduces the grid-size while ex-
    pands the filter banks.
    """
    def __init__(self,in_channels, out3x3, out2_3x3):
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out3x3, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out3x3, kernel_size=3, stride=2, padding=1,bias=False),
            )
        self.branch2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out2_3x3, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(out2_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out2_3x3, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out2_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out2_3x3, out_channels=out2_3x3, kernel_size=3, stride=2, padding=1,bias=False),
            )
        self.bn = nn.BatchNorm2d(out3x3+out2_3x3+in_channels)
        
    def forward(self, x):
        branch_1 = self.maxpool(x)
        branch_2 = self.branch3x3(x)
        branch_3 = self.branch2_3x3(x)
        output = F.relu(self.bn(torch.cat((branch_1, branch_2, branch_3), 1)))
        return output
    
