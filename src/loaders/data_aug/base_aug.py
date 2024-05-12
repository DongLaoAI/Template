# Author: Nguyen Y Hop

import cv2
import random
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


class BaseAug:

    def __init__(self, aug_thresh, **kwargs):
        self.aug_thresh = aug_thresh
        
        
    def __call__(self, data):
        img = data['img']
        data['img'] = img
        return data
    

    def test_aug(self, img):
        img = self.ExpandScaleY(image=img)
        return img





