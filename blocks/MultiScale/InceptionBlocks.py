import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    """
    This block takes an input with in_channels and applies 4 different
    convolutional layers to it. The output is the concatenation of the
    4 outputs.
    """

    def __init__(self,in_channels):
        super(InceptionBlock, self).__init__()
        
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