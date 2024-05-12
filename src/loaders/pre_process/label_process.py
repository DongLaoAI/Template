
import torch
import numpy as np


class LabelEncoder(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 **kwargs):
        pass


    def __call__(self, data):
        label = data['label']
        data['label'] = label
        return data