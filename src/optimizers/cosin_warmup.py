import math
import torch


class CosinWarmUp(object):
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, **kwargs):
        self.optimizer = optimizer
        self._step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        # self.lr_list = []

    def rsqrt(self, x):
        return 1.0 / math.sqrt(x)

    def step(self):
        self._step += 1
        arg1 = self.rsqrt(self._step)
        arg2 = self._step * (self.warmup_steps ** -1.5)

        lr = self.rsqrt(self.d_model) * min(arg1, arg2)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            # self.lr_list.append(lr)



if __name__ == '__main__':
        import torch
        import yaml
        import matplotlib.pyplot as plt
        from src.models.architectures.basemodel import BaseModel

        cfg_path = 'configs/khmer_config_lmdb_resnet_svtr_light.yml'
        cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
        model = BaseModel(cfg['Architecture'])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = CosinWarmUp(optimizer, d_model=1024, warmup_steps=4000)


        for epoch in range(10):
            for i in range(1250):
                optimizer.step()
                scheduler.step()
            plt.figure(figsize=(24,8))
            plt.plot(scheduler.lr_list)
            plt.savefig("debugs/learning_rate/0001.jpg")
