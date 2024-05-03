# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from model.CNNBaseModel import CNNBaseModel
from CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock, InceptionBlock, SqueezeAndExciteBlock


class YourNet(CNNBaseModel):

    def __init__(self, num_classes=10):
        super(YourNet, self).__init__()
        self.conv_layers = nn.Sequential(
            #32x32x3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            #32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #16x16x16
            DenseBlock(16),
            #16x16x32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0), 
            #8x8x32
            ResidualBlock(32, 64), 
            InceptionBlock(64),
            #8x8x120
            SqueezeAndExciteBlock(120,16),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            #1x1x512
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(960, num_classes),
        )

        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc_layers(x)
        return x
