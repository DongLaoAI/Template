# Author: Nguyen Y Hop
import cv2
import time
import torch
import numpy as np

class LoadImg:

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img_path = data['img_path']
        img = cv2.imread(img_path)
        data['img'] = img
        return data

class DecodeBufferImage:

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['img']
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if min(img.shape[:2]) < 4:
            return None
        data['img'] = img
        return data

class NormalizeImgAndTranpose:

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['img']
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        data['img'] = torch.tensor(img)
        return data
    

class ResizeWithPad:

    def __init__(self, shape, **kwargs):
        self.target_h, self.target_w = shape
        self.padding_color = (0, 0, 0)

    def resize_with_height(self, img, ratio, ori_w, target_height):
        new_w = int(ori_w * ratio)
        img = cv2.resize(img, (new_w, target_height))
        return img, new_w
    
    def resize_with_width(self, img, ratio, ori_h, target_width):
        new_h = int(ori_h * ratio)
        img = cv2.resize(img, (target_width, new_h))
        return img, new_h

    def __call__(self, data):
        img = data['img']
        ori_h, ori_w, _ = img.shape
        h_ratio = self.target_h / ori_h
        w_ratio = self.target_w / ori_w
        top, bottom, left, right = 0, 0, 0, 0

        if h_ratio > 1.0 and w_ratio > 1.0:
            top = (self.target_h - ori_h) // 2
            bottom = (self.target_h - ori_h) - top
            left = (self.target_w - ori_w) // 2
            right = (self.target_w - ori_w) - left

        else:
            if h_ratio > w_ratio:
                img, _ = self.resize_with_width(img, w_ratio, ori_h, self.target_w)
            else:
                img, _ = self.resize_with_height(img, h_ratio, ori_w, self.target_h)
            ori_h, ori_w, _ = img.shape
            top = (self.target_h - ori_h) // 2
            bottom = (self.target_h - ori_h) - top
            left = (self.target_w - ori_w) // 2
            right = (self.target_w - ori_w) - left

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )
        data['img'] = img
        return data
    