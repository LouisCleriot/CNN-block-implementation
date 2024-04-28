import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarshicalSplitBlock(nn.Module):
    """We split the input into S groups channel-wise,
    first group is directly passed to the output, the second
    through a 3x3 bn relu, his output is split in 2, the first
    half is concatenated with next group and the second half is
    passed to the output. The process is repeated until the last

    Args:
        s (int): the number of groups to split the input
        in_channels (int): the number of input channels
    """
    
    def __init__(self, s, in_channels):
        super(HierarshicalSplitBlock, self).__init__()
        self.s = s
        group_size = in_channels // s
        self.groups = nn.ModuleList()
        added_channels = 0
        for i in range(1, s):
            in_channels = group_size + added_channels
            self.groups.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))
            added_channels = in_channels//2

    def forward(self, x):
        groups = torch.chunk(x, self.s, 1)
        output = [groups[0]]
        for i in range(1, self.s):
            group = self.groups[i-1](groups[i])
            if i == self.s-1:
                output.append(group)
                break
            group1, group2 = torch.chunk(group, 2, 1)
            output.append(group2)
            groups[i+1] = torch.cat((groups[i+1], group1), 1)
        return output
            
        