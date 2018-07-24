import torch
from torch import nn
import torch.nn.functional as F
from settings import BLOCKS

def conv3x3(inchannels, channels, ks=3, s=1, p=1):
    return nn.Conv2d(inchannels, channels, kernel_size=ks, stride=s, padding=p)
def conv1x1(inchannels, channels, ks=1, s=1, p=0):
    return nn.Conv2d(inchannels, channels, kernel_size=ks, stride=s, padding=p)
class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

class ResnetBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.b1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.b2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.b1(out))
        out = self.conv2(out)
        out = self.b2(out)
        out_x = F.relu(out + x)
        return out_x

class MainResnet(nn.Module):
    def __init__(self, create_block, inchannels=2, channels=64, blocks=30):
        super().__init__()        
        self.conv1 = conv3x3(inchannels, channels)
        self.b1 = nn.BatchNorm2d(channels)
        self.resnetblock = self._make_blocks(channels, create_block, blocks)
        
    def _make_blocks(self, channels, resnet_block, blocks):
        layers = []
        for i in range(0, blocks):
            layers.append(resnet_block(channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnetblock(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, inchannels=64, board_size=8):
        super().__init__()
        self.conv1 = conv1x1(inchannels, 1)
        self.b = nn.BatchNorm2d(1)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(board_size**2, 256)
        self.linear2 = nn.Linear(256, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.b(x))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.tanh(self.linear2(x))
        return x

class PolicyHead(nn.Module):
    def __init__(self, inchannels=64, board_size=8, vocab=512):
        super().__init__()
        self.conv1 = conv1x1(inchannels, 1)
        self.b = nn.BatchNorm2d(1)
        self.flatten = Flatten()
        self.linear = nn.Linear(board_size**2, 512)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.b(x))
        x = self.flatten(x)
        x = self.logsoftmax(self.linear(x))
        return x

class GolaiZero(nn.Module):
    def __init__(self, inchannels=2, channels=64, board_size=8, vocab=512, blocks=30):
        super().__init__()
        self.resnet = MainResnet(ResnetBlock, inchannels, channels)
        self.policyhead = PolicyHead(channels, board_size, vocab)
        self.valuehead = ValueHead(channels, board_size)
    def forward(self, x):
        features = self.resnet(x)
        policy_out = self.policyhead(features)
        value_out = self.valuehead(features)
        return policy_out, value_out


# Print overview of model
# from torchsummary import summary
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# golai_zero = GolaiZero().to(device)
