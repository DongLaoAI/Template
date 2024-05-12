import torch
from torch import nn
import  torch.nn.functional as F


class CTCHead(nn.Module):

    def __init__(self, in_channels, num_classes, device='cpu', **kwargs):
        super().__init__()
        self._fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = self._fc(x)
        return x

if __name__ == '__main__':
    net = CTCHead(128, 96)
    inputs = torch.randn([2, 128, 1, 200])
    outputs = net(inputs)
    print(outputs.shape)