# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock, SimpleCNNBlock


class YourNet(nn.Module):

    def __init__(self, num_classes=10):
        super(YourNet, self).__init__()
        self.conv_layers = nn.Sequential(
            #32x32x3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            #32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16,32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            #16x16x32
            DenseBlock(in_channels=32),
            #16x16x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #8x8x64
            DenseBlock(in_channels=64),
            #8x8x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DenseBlock(in_channels=128),
            BottleneckBlock(in_channels=256, downsample=4),
            #8x8x256
            nn.Dropout(p=0.2),
            ResidualBlock(256,128),
            BottleneckBlock(in_channels=128, downsample=4),
            #4x4x128
            nn.AdaptiveAvgPool2d(1),
            #1x1x128
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc_layers(x)
        return x
