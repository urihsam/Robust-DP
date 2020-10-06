from dependency import *
from tensorflow.python import pywrap_tensorflow
import os


def ckpt_to_list(path):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map =  reader.get_variable_to_shape_map()
    return list(var_to_shape_map.keys())


def ckpt_to_dict(path):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map =  reader.get_variable_to_shape_map()
    ckpt_dict = {}
    for key in var_to_shape_map:
        print(key)
        ckpt_dict[key] = reader.get_tensor(key)
    return ckpt_dict


"""def ckpt_to_dict(path):
    reader = tf.train.NewCheckpointReader(path)
    restore_dict = dict()
    for v in tf.trainable_variables():
        print(v.name)
        tensor_name = v.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            print('has tensor ', tensor_name)
            restore_dict[tensor_name] = v
    return restore_dict"""
