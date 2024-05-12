# Author: Nguyen Y Hop
import os
import yaml
import torch
import itertools
from tqdm import tqdm

from torch import nn
from torch.optim import Adam
from terminaltables import AsciiTable
from torch.utils.data import DataLoader


from src.losses import *
from src.optimizers import *
from src.models.architectures import *

from src.loaders.stream_data import *
from src.losses import *
from tools.utils import *
from src.loaders.post_process import *
from evals.evaluate_func import *


class Trainer:

    def __init__(self, cfg):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_cfg = cfg['Global']
        self.model_cfg = cfg['Architecture']
        self.optimizer_cfg = cfg['Optimizer']
        self.loader_cfg = cfg['Data']
        self.criterion_cfg = cfg['Criterion']
        self.pretrain_cfg = cfg['Pretrain']
        self.post_process_cfg = cfg['PostProcess']

        self.epoch = self.global_cfg['epoch']
        self.batch_size = self.global_cfg['batch_size']

        self.prepare_structure()
        
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.build_scheduler()
        self.train_dataloader = self.build_loader(mode='Train', batch_size=self.batch_size)
        self.test_dataloader = self.build_loader(mode='Valid', shuffle=False, batch_size=16)
        self.criterion = self.build_criterion()

        self.load_pretrain()

        
    def prepare_structure(self):

        checkpoint_dir = os.path.join(self.global_cfg['checkpoint'])
        self.epoch_ckpt_dir = os.path.join(checkpoint_dir, 'epoch')
        self.iter_acc_ckpt_dir = os.path.join(checkpoint_dir, 'iter_acc')
        self.best_acc_ckpt_dir = os.path.join(checkpoint_dir, 'best_acc')

        self.log_path = self.global_cfg['log_path']
        self.log_dir = "/".join(self.log_path.split('/')[:-1])

        os.makedirs(self.epoch_ckpt_dir, exist_ok=True)
        os.makedirs(self.iter_acc_ckpt_dir, exist_ok=True)
        os.makedirs(self.best_acc_ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)



    def build_model(self):
        name = self.model_cfg['name']
        model = eval(name)(self.model_cfg, device=self.device)
        return model
    

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        name = self.optimizer_cfg['name']
        if name == 'Adam':
            optimizer = torch.optim.Adam(
                params=params,
                lr=self.optimizer_cfg['lr'])
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=0.1)
        else:
            raise Exception(f'{name} optimzier does not exist')
        return optimizer

    def build_scheduler(self):
        if 'Scheduler' in self.optimizer_cfg:
            schedule_cfg = self.optimizer_cfg['Scheduler']
            self.scheduler = eval(schedule_cfg['name'])(optimizer=self.optimizer, **schedule_cfg)
        else:
            self.scheduler = None
    
        
    def build_loader(self, mode='Train', shuffle=True, batch_size=32,):
        name = self.loader_cfg[mode]['name']
        dataset = eval(name)(self.loader_cfg[mode])  
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        return data_loader
    

    def build_criterion(self):
        name = self.criterion_cfg['name']
        criterion = eval(name)(self.criterion_cfg)  
        return criterion
    

    def compute_loss(self, predicts, targets):
        losses = self.criterion(predicts, targets)
        return losses
    

    def load_pretrain(self):
        if self.pretrain_cfg['resume'] and self.pretrain_cfg['checkpoint_path'] is not None:
            if not os.path.exists(self.pretrain_cfg['checkpoint_path']):
                print('Cannot find weight path to load')
                return False
            print(f'\n-----Load pretrain-----')
            checkpoint = torch.load(self.pretrain_cfg['checkpoint_path'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.ckpt_epoch = checkpoint['epoch']
    

    def train(self):
        log = open(self.log_path, 'w')
        log.write('\n=======================TRAINING===============================')

        iteration = 0
        avg_loss = 0
        iter_eval = self.global_cfg['iter_eval']
        
        self.ckpt_epoch = 0
        best_acc = 0.0

        self.model.train()
        self.model.to(self.device)
     
        print('\n-----Training model-----')
        for epoch in range(1 + self.ckpt_epoch, self.epoch + 1):

            self.train_dataloader.dataset.shuffle_data()

            for samples in tqdm(self.train_dataloader):
    
                pass


if __name__ == '__main__':
    cfg_path = 'configs/cfg.yml'
    cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
    trainer = Trainer(cfg)
    trainer.train()

    """
    python3 -m trainer.train
    """
    
