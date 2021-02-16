
# coding: utf-8

# In[1]:
from PIL import Image
from tensorflow.python.platform import gfile
from dependency import *
import tensorflow as tf
import numpy as np
import random
import gzip
import os
import skimage
import skimage.io
import skimage.transform


# ## Training images

# In[2]:

class dataset(object):
    """
    This data class is designed for mnist dataset, which has three different kinds of data --
    train, val and test. I split training dataset (60000) into train (54000) and valid (6000)
    datasets by the split_ratio
    
    """
    def __init__(self):
        print("Dataset here")
        (x_train_, y_train_), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train_ = x_train_.astype('float32') 
        x_test = x_test.astype('float32') 
        # resize
        #x_train_ = np.array(self.resize_data(x_train_, (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
        #x_test = np.array(self.resize_data(x_test, (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS)))
        #import pdv; pdb.set_trace()
        # one hot
        y_train_ = np.array(self._one_hot_encode(y_train_, 10))
        y_test = np.array(self._one_hot_encode(y_test, 10))

        #
        total_size = x_train_.shape[0]
        self.train_size = int(total_size * 0.8)
        self.valid_size = total_size - self.train_size
        self.test_size = x_test.shape[0]

        import random
        # shuffling
        total_idx = list(range(total_size))
        random.shuffle(total_idx)
        train_idx = total_idx[:self.train_size]
        self.x_train = x_train_[train_idx]
        self.y_train = y_train_[train_idx]
        #
        valid_idx = total_idx[self.train_size:]
        self.x_valid = x_train_[valid_idx]
        self.y_valid = y_train_[valid_idx]
        #
        test_idx = list(range(self.test_size))
        random.shuffle(test_idx)
        self.x_test = x_test[test_idx]
        self.y_test = y_test[test_idx]
    
    # data
    def _one_hot_encode(self, inputs, encoded_size):
        def get_one_hot(number):
            on_hot=[0]*encoded_size
            on_hot[int(number)]=1
            return on_hot
        return list(map(get_one_hot, inputs))

    def resize_data(self, inputs, shape):
        def resize(in_):
            return skimage.transform.resize(in_, shape, mode='constant')
        return list(map(resize, inputs))

    def shuffle_train(self):
        # shuffling
        train_idx = list(range(self.train_size))
        random.shuffle(train_idx)
        self.x_train = self.x_train[train_idx]
        self.y_train = self.y_train[train_idx]

    def shuffle_valid(self):
        # shuffling
        valid_idx = list(range(self.valid_size))
        random.shuffle(valid_idx)
        self.x_valid = self.x_valid[valid_idx]
        self.y_valid = self.y_valid[valid_idx]

    def shuffle_test(self):
        # shuffling
        test_idx = list(range(self.test_size))
        random.shuffle(test_idx)
        self.x_test = self.x_test[test_idx]
        self.y_test = self.y_test[test_idx]


