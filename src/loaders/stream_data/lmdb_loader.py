# Author: Nguyen Y Hop
import os
import cv2
import lmdb
import yaml
import random
import numpy as np
from torch.utils.data import Dataset

from ..utils.transform_utils import *
from ..pre_process.img_process import *
from ..data_aug.base_aug import *



class LMDBDataset(Dataset):

    def __init__(self, cfg, **kwargs):

        self.cfg = cfg
        self.data_dir_list = self.cfg['data_dir']
        self.parse_data()
        self.transform_funcs = init_transforms(self.cfg['Transforms'])
        self.keep_keys = self.cfg['Keepkeys']

    def parse_data(self):
        self.envs = []
        self.idx_matrixes = []
        env_id = 0

        for data_dir in self.data_dir_list:

            for lmdb_dir, _, path_ in os.walk(data_dir):
                if len(path_) == 0:
                    continue
                
                print(">>>>>>>>>>", lmdb_dir)
                env = lmdb.open(lmdb_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
                self.envs.append(env)

                with env.begin(write=False) as txn:
                    nSamples = int(txn.get('num-samples'.encode()))
                    idx_matrix = np.zeros((nSamples, 2), dtype=np.int32)

                    for index in range(nSamples):
                        idx_matrix[index] = [env_id, index]
                    self.idx_matrixes.append(idx_matrix)

                env_id += 1

        self.idx_matrixes = np.concatenate(self.idx_matrixes, axis=0)
        print('Num sample:\t', self.idx_matrixes.shape[0])

    def shuffle_data(self):
        idx_matrixes = self.idx_matrixes.tolist()
        random.shuffle(idx_matrixes)
        self.idx_matrixes = np.array(idx_matrixes)


    def __len__(self):
        return self.idx_matrixes.shape[0]

    def __getitem__(self, index):
        env_id, data_id = self.idx_matrixes[index]
        data = {}
        if True:
            with self.envs[env_id].begin(write=False) as txn:
                label_key = 'label-%09d'.encode() % data_id
                label = txn.get(label_key).decode('utf-8')
                img_key = 'image-%09d'.encode() % data_id
                imgbuf = txn.get(img_key)
                data['img'] = imgbuf
                data['text'] = label.strip()
            if self.transform_funcs is not None:
                data = transform_processing(data, self.transform_funcs)
            data = keep_keys(data, self.keep_keys)
        else:
            data = self.__getitem__(np.random.randint(0, self.idx_matrixes.shape[0]-1))
        if data is None:
            data = self.__getitem__(np.random.randint(0, self.idx_matrixes.shape[0]-1))
        return data



        

