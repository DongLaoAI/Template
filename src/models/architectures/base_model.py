import torch
from torch import nn

from ..backbones import *
from ..necks import *
from ..heads import *


class BaseModel(nn.Module):

    def __init__(self, cfg, device='cuda', **kwargs):
        super().__init__()
        
        backbone_cfg = cfg['Backbone']
        neck_cfg = cfg.get('Neck')
        head_cfg = cfg.get('CTCHead')

        self.backbone = eval(backbone_cfg['name'])(**backbone_cfg, device=device)

        self.neck = None
        if neck_cfg is not None:
            self.neck = eval(neck_cfg['name'])(**neck_cfg, device=device)

        self.ctc_head = None
        if head_cfg is not None:
            self.head = eval(head_cfg['name'])(**head_cfg, device=device)

    def inference(self, inputs):
        x = self.backbone(inputs)
        x = self.neck(x)
        x = self.head(x)
        return x

    def forward(self, inputs):
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        if self.head is not None:
            x = self.head(x)
        return x

if __name__ == '__main__':
    import yaml
    cfg = yaml.load(open('configs/cambidian_config_lmdb.yml'), yaml.Loader)['Architecture']
    net = BaseModel(cfg)
    inputs = torch.randn((2, 3, 42, 800))
    outputs = net(inputs)
    print(outputs.shape)

