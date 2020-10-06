
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


# ## Training images

# In[2]:

class dataset(object):
    """
    This data class is designed for mnist dataset, which has three different kinds of data --
    train, val and test. I split training dataset (60000) into train (54000) and valid (6000)
    datasets by the split_ratio
    
    """
    def __init__(self, data_dir, image_dir="images", split_ratio=0.9, 
                 onehot=True, normalize=True, biased=True, 
                 adv_path_prefix="fgsm"):
        print("Dataset here")
        self.data_dir = data_dir
        self.train_dir = self.path("train")
        self.test_dir = self.path("test")
        self.image_dir = image_dir
        self.split_ratio = split_ratio
        # It's a switch when init dataset, default as off, if on, then dataset only store image paths 
        # and class names; otherwise, the dataset store data directly
        self.onehot = onehot
        self.normalize = normalize
        self.biased = biased
        self.adv_path_prefix = adv_path_prefix
        # Obtain two lists of tuples in format (image file path, image class name)
        self._train_image_names_classes, self._valid_image_names_classes = self._get_train_valid_img_names_and_classes()
        # Obtain a list of tuples in format (image file path, image class name)
        self._test_image_names_classes = self._get_test_img_names_and_classes(split_ratio)
        self.shuffle() # shuffle the lists
       
    
    def path(self, *path):
        return os.path.join(self.data_dir, *path)
    
    
     # One hot encoding
    def _one_hot_encode(self, inputs, encoded_size):
        def get_one_hot(number):
            on_hot=[0]*encoded_size
            on_hot[int(number)]=1
            return on_hot
        #return list(map(get_one_hot, inputs))
        if isinstance(inputs, list):
            return list(map(get_one_hot, inputs))
        else:
            return get_one_hot(inputs)
    
    @property
    def num_classes(self): return FLAGS.NUM_CLASSES
    
    @property
    def encoding_dict(self): return self._encoding_dict
    
    # Obtain two lists of tuples in format (image file path, image class name)
    def _get_train_valid_img_names_and_classes(self):
        fuse = lambda img, cls: list(zip(img, cls))
        train_img_names_classes_list = []
        for class_idx in range(FLAGS.NUM_CLASSES):
            class_name = str(class_idx)
            data_dir = self.train_dir + "/" + class_name + "/" + self.image_dir
            train_img_names_list = tf.gfile.Glob(os.path.join(data_dir, '*.npy'))
            train_class_names_list = [class_name] * len(train_img_names_list)
            train_img_names_classes_list += fuse(train_img_names_list, train_class_names_list)
        random.shuffle(train_img_names_classes_list)
        total_size = len(train_img_names_classes_list)
        train_size = int(total_size * self.split_ratio)
        return train_img_names_classes_list[:train_size], train_img_names_classes_list[train_size:]
    
    def _get_test_img_names_and_classes(self, split_ratio):
        fuse = lambda img, cls: list(zip(img, cls))
        test_img_names_classes_list = []
        for class_idx in range(FLAGS.NUM_CLASSES):
            class_name = str(class_idx)
            data_dir = self.test_dir + "/" + class_name + "/" + self.image_dir
            test_img_names_list = tf.gfile.Glob(os.path.join(data_dir, '*.npy'))
            test_class_names_list = [class_name] * len(test_img_names_list)
            test_img_names_classes_list += fuse(test_img_names_list, test_class_names_list)
        return test_img_names_classes_list
    
    # Load images using the list of tuple in format (image file path, image class name)
    def _load_image(self, img_name_class, onehot, normalize, biased, is_np=True):
        file_path, class_name = img_name_class
        path = os.path.join(file_path)
        if is_np: # np file
            try:
                image = np.load(path)
            except:
                return None
        else:
            im = Image.open(path)
            image = np.asarray(im)
            if FLAGS.NUM_CHANNELS == 1:
                image = np.expand_dims(image, axis=2)
            if image.shape != (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS):
                # e.g. grayscale
                return None
            assert image.dtype == np.uint8
            image = image.astype(np.float32)
        assert image.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
        # print(image.shape, file_path)
        if normalize:
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = image / 255.0 
            if biased:
                image = image * 2.0 - 1.0

        # labels
        label = int(class_name)
        #print(file_path, class_name, label)
        #print(label)
        if onehot:
            label = self._one_hot_encode(label, self.num_classes)
        #print(label)
        return (image, label)
    
    @property
    def train_size(self): return len(self._train_image_names_classes)
    
    @property
    def valid_size(self): return len(self._valid_image_names_classes)
    
    @property
    def test_size(self): return len(self._test_image_names_classes)
    
    def shuffle(self): 
        self.train_shuffle()
        self.valid_shuffle()
        self.test_shuffle()
        
    def train_shuffle(self): 
        random.shuffle(self._train_image_names_classes)
        self._train_batch_idx = 0
    
    def valid_shuffle(self):
        random.shuffle(self._valid_image_names_classes)
        self._valid_batch_idx = 0
        
    def test_shuffle(self): 
        random.shuffle(self._test_image_names_classes)
        self._test_batch_idx = 0
    
    def next_train_batch(self, batch_size, with_path=False):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, FLAGS.NUM_CLASSES], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        paths = []
        idx = 0
        while idx < batch_size:
            if self._train_batch_idx >= self.train_size:
                self.train_shuffle()
            img_name_class = self._train_image_names_classes[self._train_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            #
            path_name = img_name_class[0].split("/")
            atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
            if res is not None:
                if os.path.exists(atk_path):
                    images[idx, :, :, :] = res[0]
                    labels[idx, :] = res[1]
                    if not with_path:
                        # atk
                        atk_res = self._load_image([atk_path, img_name_class[1]], self.onehot, self.normalize, self.biased)
                        atk_images[idx, :, :, :] = atk_res[0]
                    #
                    idx += 1
                #  
                paths.append(img_name_class[0])
                
            self._train_batch_idx += 1
        if with_path:
            return images, labels, paths
        else:
            return images, labels, atk_images
    
    def next_valid_batch(self, batch_size, with_path=False):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, FLAGS.NUM_CLASSES], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        paths = []
        idx = 0
        while idx < batch_size:
            if self._valid_batch_idx >= self.valid_size:
                self.valid_shuffle()
            img_name_class = self._valid_image_names_classes[self._valid_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            #
            path_name = img_name_class[0].split("/")
            atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
            if res is not None:
                if os.path.exists(atk_path):
                    images[idx, :, :, :] = res[0]
                    labels[idx, :] = res[1]
                    if not with_path:
                        # atk
                        atk_res = self._load_image([atk_path, img_name_class[1]], self.onehot, self.normalize, self.biased)
                        atk_images[idx, :, :, :] = atk_res[0]
                    #
                    idx += 1
                #  
                paths.append(img_name_class[0])
            self._valid_batch_idx += 1
        if with_path:
            return np.array(images), np.array(labels), paths
        else:
            return np.array(images), np.array(labels), atk_images
    
    def next_test_batch(self, batch_size, with_path=False):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, FLAGS.NUM_CLASSES], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        idx = 0
        paths = []
        while idx < batch_size:
            if self._test_batch_idx >= self.test_size:
                self.test_shuffle()
            img_name_class = self._test_image_names_classes[self._test_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            #
            path_name = img_name_class[0].split("/")
            atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
            if res is not None:
                if os.path.exists(atk_path):
                    images[idx, :, :, :] = res[0]
                    labels[idx, :] = res[1]
                    if not with_path:
                        # atk
                        atk_res = self._load_image([atk_path, img_name_class[1]], self.onehot, self.normalize, self.biased)
                        atk_images[idx, :, :, :] = atk_res[0]
                    #
                    idx += 1
                #  
                paths.append(img_name_class[0])
            self._test_batch_idx += 1
        if with_path:
            return np.array(images), np.array(labels), paths
        else:
            return np.array(images), np.array(labels), atk_images

