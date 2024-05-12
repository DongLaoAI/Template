# Author: Nguyen Y Hop
from ..pre_process.img_process import *
from ..pre_process.label_process import *
from ..data_aug.base_aug import *

def keep_keys(data, keep_keys):
    output = []
    for key in keep_keys:
        output.append(data[key])
    return output


def init_transforms(trans_cfg):
    module_list = []
    for dict_ in trans_cfg:
        params = list(dict_.values())[0]
        name = params['name']
        module = eval(name)(**params)
        module_list.append(module)
    return module_list


def transform_processing(data, trans_funcs):
    if len(trans_funcs) == 0:
        return data
    for module in trans_funcs:
        data = module(data)
    return data