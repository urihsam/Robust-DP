
# coding: utf-8

# In[1]:

from PIL import Image
from dependency import *
import tensorflow as tf
import numpy as np
import random
import os


# ## Training images

# In[2]:

class dataset(object):
    """
    This data class is designed for imagent dataset, which has three different knids of data --
    train, val and test. The file structures of these datasets are different, what's worse, the test 
    dataset doesn't contain any labels.
    Here, I split the validation dataset, which used to have 50,000 images, into validation (10,000) and 
    test (40,000) datasets.
    
    """
    def __init__(self, data_dir, split_ratio=0.8, 
                 onehot=True, normalize=True, biased=True):
        print("Dataset here")
        self.data_dir = data_dir
        self.train_dir = self.path("train")
        self.test_dir = self.path("valid")
        #self.image_dir = image_dir
        # It's a switch when init dataset, default as off, if on, then dataset only store image paths 
        # and class names; otherwise, the dataset store data directly
        self.onehot = onehot
        self.normalize = normalize
        self.biased = biased

        # Build an encoding dictionary to map from class names to integer label
        self._encoding_dict = np.load(self.path("class_name_id_dict.npy")).item()
        # Load the names of all classes
        self._class_names = self._encoding_dict.keys()
        # Obtain a list of tuples in format (image file path, image class id)
        self._train_image_names_classes = self._get_train_img_names_and_class_ids()
        # Obtain two lists of tuples in format (image file path, image class id)
        self._valid_image_names_classes, self._test_image_names_classes =  self._get_valid_test_img_names_and_class_ids(split_ratio)
        self.shuffle() # shuffle the lists
       
    
    def path(self, *path):
        return os.path.join(self.data_dir, *path)
    
     # One hot encoding
    def _one_hot_encode(self, inputs, encoded_size):
        def get_one_hot(number):
            on_hot=[0]*encoded_size
            on_hot[int(number)-1]=1
            return on_hot
        #return list(map(get_one_hot, inputs))
        if isinstance(inputs, list):
            return list(map(get_one_hot, inputs))
        else:
            return get_one_hot(inputs)
    
    @property
    def class_names(self): return self._class_names
    
    @property
    def num_classes(self): return len(self._class_names)
    
    @property
    def encoding_dict(self): return self._encoding_dict
    
    # Obtain a list of tuples in format (image file path, image class id)
    def _get_train_img_names_and_class_ids(self):
        fuse = lambda img, ids: list(zip(img, ids))
        train_img_names_classes_list = []
        for class_name in self._class_names:
            data_dir = self.train_dir + "/images/" + class_name
            train_img_names_list = tf.gfile.Glob(os.path.join(data_dir, '*.JPEG'))
            random.shuffle(train_img_names_list)

            '''
            if len(train_img_names_list) > 800:
                train_img_names_list = train_img_names_list[:800]
            '''
            train_class_ids_list = [self._encoding_dict[class_name]] * len(train_img_names_list)
            train_img_names_classes_list += fuse(train_img_names_list, train_class_ids_list)
        return train_img_names_classes_list
    
    def _get_valid_test_img_names_and_class_ids(self, split_ratio):
        fuse = lambda img, ids: list(zip(img, ids))
        test_class_ids_list = np.load(os.path.join(self.test_dir, "val_class_ids.npy"))
        test_img_names_list = tf.gfile.Glob(os.path.join(self.test_dir, "images", "*.JPEG"))
        test_img_names_classes_list = []
        for test_img_names in test_img_names_list:
            img_full_name = test_img_names.split("/")[-1] # X_X_X_X.JPEG
            img_name = img_full_name.split(".")[0] #X_X_X_X
            idx = int(img_name.split("_")[-1]) - 1 # X
            #if test_class_ids_list[idx] <= 200:
            test_img_names_classes_list.append(tuple([test_img_names, test_class_ids_list[idx]]))
        # Shuffle before splitting
        random.shuffle(test_img_names_classes_list)
        total_size = len(test_img_names_classes_list)
        valid_size = int(total_size * split_ratio)
        return test_img_names_classes_list[: valid_size], test_img_names_classes_list[valid_size: ]
    
    # Load images using the list of tuple in format (image file path, image class name)
    def _load_image(self, img_name_class, onehot, normalize, biased):
        file_path, class_id = img_name_class
        path = os.path.join(file_path)
        im = Image.open(path)
        image = np.asarray(im)
        if image.shape != (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS):
            print ("Wrong image shape: {}".format(image.shape))
            # e.g. grayscale
            return None
        #assert image.dtype == np.uint8
        image = image.astype(np.float32)
        assert image.shape == (FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)
        # print(image.shape, file_path)
        if normalize:
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = image / 255.0 
            if biased:
                image = image * 2.0 - 1.0

        # labels
        label = int(class_id)
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
    
    def next_train_batch(self, batch_size):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, self.num_classes], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        idx = 0
        while idx < batch_size:
            if self._train_batch_idx >= self.train_size:
                self.train_shuffle()
            img_name_class = self._train_image_names_classes[self._train_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                idx += 1
            self._train_batch_idx += 1
        
        return np.array(images), np.array(labels)
    
    def next_valid_batch(self, batch_size):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, self.num_classes], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        idx = 0
        while idx < batch_size:
            if self._valid_batch_idx >= self.valid_size:
                self.valid_shuffle()
            img_name_class = self._valid_image_names_classes[self._valid_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                idx += 1
            self._valid_batch_idx += 1
        
        return np.array(images), np.array(labels)
    
    def next_test_batch(self, batch_size):
        images = np.ndarray([batch_size, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, self.num_classes], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        idx = 0
        while idx < batch_size:
            if self._test_batch_idx >= self.test_size:
                self.test_shuffle()
            img_name_class = self._test_image_names_classes[self._test_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                idx += 1
            self._test_batch_idx += 1
        
        return np.array(images), np.array(labels)

