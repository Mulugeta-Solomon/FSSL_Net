import torch 
import torch.nn as nn
import torch.nn.functional as F 

__all__ = ["HourglassNet", "hg"]


class Bottleneck2D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out 
    
class Hourglass(nn.Module):
    def __init__(self, block, num_block, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_block, planes, depth)

    def _make_residual(self, block, num_blocks, planes, depth):
        layers = []
        
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers)

    
        


    