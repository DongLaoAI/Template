import torch
from torch import nn

class Img2Seq(nn.Module):
    def __init__(self, in_channels=512, device='cuda', **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.squeeze(axis=2)
        x = x.permute([0, 2, 1]) 
        return x


if __name__ == '__main__':
    inputs = torch.randn((2, 512, 1, 200))
    net = Img2Seq()
    outputs = net(inputs)
    print(outputs.shape)