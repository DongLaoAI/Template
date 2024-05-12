import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):

    def __init__(self, in_channels ,out_channels, stride=1, padding=(1, 1)):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=stride,
                               padding=padding)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        if stride != 1:
            layer_downsample = []
            layer_downsample.append(('conv_down', nn.Conv2d(in_channels=in_channels,
                                                            out_channels=out_channels,
                                                            kernel_size=(1, 1),
                                                            stride=stride)))
            layer_downsample.append(('bn_down',nn.BatchNorm2d(num_features=out_channels)))
            self.downsample = nn.Sequential(OrderedDict(layer_downsample))
        else:
            self.downsample = lambda x: x

    def forward(self, inputs):
        residual = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = F.relu(residual + x)
        return output



def make_basic_block_layer(in_channels, out_channels, blocks, stride=1):
    res_block = []
    res_block.append(('basic_block_0', BasicBlock(in_channels, out_channels, stride=stride)))

    for i in range(1, blocks):
        res_block.append((f'basic_block_{i}', BasicBlock(out_channels, out_channels, stride=1)))

    return nn.Sequential(OrderedDict(res_block))


class ResNetTypeI(nn.Module):
    def __init__(self, out_channel=512, layer_params=[2, 2, 2, 2], **kwargs):
        super(ResNetTypeI, self).__init__()

        self.conv1 = nn.Conv2d( in_channels=3,
                                out_channels=32,
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3))

        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3),
                                  stride=1,
                                  padding=(1, 1))

        self.layer1 = make_basic_block_layer(in_channels=32,
                                             out_channels=64,
                                             blocks=layer_params[0],
                                             stride=(2, 1))

        self.layer2 = make_basic_block_layer(in_channels=64,
                                             out_channels=128,
                                             blocks=layer_params[1],
                                             stride=(2, 2))

        self.conv_2 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=(3, 1),
                                stride=(2, 2))

        self.layer3 = make_basic_block_layer(in_channels=128,
                                             out_channels=256,
                                             blocks=layer_params[2],
                                             stride=(2, 1))

        self.layer4 = make_basic_block_layer(in_channels=256,
                                             out_channels=out_channel,
                                             blocks=layer_params[3],
                                             stride=(2, 1))

        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



if __name__ == '__main__':
    import time
    net = ResNetTypeI(layer_params=[2, 2, 2, 2])
    inputs = torch.randn((2, 3, 42, 800))
    st = time.time()
    for i in range(0, 10):
        outputs = net(inputs)
        print(outputs.shape)
    et = time.time()
    print(et - st)
    