
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
    This data class is designed for tiny-imagent dataset, which has three different knids of data --
    train, val and test. The file structures of these datasets are different, what's worse, the test 
    dataset doesn't contain any labels.
    Here, I split the validation dataset, which used to have 10,000 images, into validation (5,000) and 
    test (5,000) datasets.
    
    """
    def __init__(self, data_dir, image_dir="images", split_ratio=0.5, 
                 onehot=True, normalize=True, biased=True,
                 adv_path_prefix="fgsm"):
        print("Dataset here")
        self.data_dir = data_dir
        self.train_dir = self.path("train")
        self.test_dir = self.path("val")
        self.image_dir = image_dir
        # It's a switch when init dataset, default as off, if on, then dataset only store image paths 
        # and class names; otherwise, the dataset store data directly
        self.onehot = onehot
        self.normalize = normalize
        self.biased = biased
        self.adv_path_prefix = adv_path_prefix
        # Load the names of all classes
        self._class_names = self._get_class_names_from_wnids()
        # Build an encoding dictionary to map from class names to integer label
        self._encoding_dict = (lambda source: dict(zip(source, list(range(len(source))))))(self._class_names)
        # Obtain two lists of tuples in format (image file path, image class name)
        self._train_image_names_classes = self._get_train_img_names_and_classes()
        # Obtain a list of tuples in format (image file path, image class name)
        self._valid_image_names_classes, self._test_image_names_classes =             self._get_valid_test_img_names_and_classes(split_ratio)
        self.shuffle() # shuffle the lists
       
    
    def path(self, *path):
        return os.path.join(self.data_dir, *path)
    
    # Load the names of all classes
    def _get_class_names_from_train_dir(self):
        class_names = tf.gfile.ListDirectory(self.train_dir) # class_names are also the folder
        for class_name in class_names:
            if class_name.startswith("."):
                class_names.remove(class_name)
        return class_names
    
    def _get_class_names_from_wnids(self):
        with open(self.path('wnids.txt')) as f:
            wnids = f.readlines()
            assert len(wnids) == 200
            wnids = [x.strip() for x in wnids]
        return wnids
    
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
    def class_names(self): return self._class_names
    
    @property
    def num_classes(self): return len(self._class_names)
    
    @property
    def encoding_dict(self): return self._encoding_dict
    
    # Obtain two lists of tuples in format (image file path, image class name)
    def _get_train_img_names_and_classes(self):
        fuse = lambda img, cls: list(zip(img, cls))
        train_img_names_classes_list = []
        valid_img_names_classes_list = []
        for class_name in self._class_names:
            data_dir = self.train_dir + "/" + class_name + "/" + self.image_dir
            train_img_names_list = tf.gfile.Glob(os.path.join(data_dir, '*.JPEG'))
            train_class_names_list = [class_name] * len(train_img_names_list)
            train_img_names_classes_list += fuse(train_img_names_list, train_class_names_list)
        return train_img_names_classes_list
    
    def _get_valid_test_img_names_and_classes(self, split_ratio):
        info_file = tf.gfile.Glob(os.path.join(self.test_dir, '*.txt'))[0]
        test_img_names_classes_list = []
        with open(info_file, "r") as f:
            for line in f:
                file_name, class_name = line.split("\t")[:2]
                file_path = self.test_dir + "/" + self.image_dir + "/" + file_name
                test_img_names_classes_list.append(tuple([file_path, class_name]))
        # Shuffle before splitting
        random.shuffle(test_img_names_classes_list)
        total_size = len(test_img_names_classes_list)
        valid_size = int(total_size * split_ratio)
        return test_img_names_classes_list[: valid_size], test_img_names_classes_list[valid_size: ]
    
    # Load images using the list of tuple in format (image file path, image class name)
    def _load_image(self, img_name_class, onehot, normalize, biased, is_np=False):
        file_path, class_name = img_name_class
        path = os.path.join(file_path)
        if is_np: # np file
            image = np.load(path)
        else:
            im = Image.open(path)
            image = np.asarray(im)
            if image.shape != (64, 64, 3):
                # e.g. grayscale
                return None
            assert image.dtype == np.uint8
            image = image.astype(np.float32)
        assert image.shape == (64, 64, 3)
        # print(image.shape, file_path)
        if normalize:
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = image / 255.0 
            if biased:
                image = image * 2.0 - 1.0

        # labels
        label = self._encoding_dict[class_name]
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
        images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, 200], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        paths = []
        idx = 0
        while idx < batch_size:
            if self._train_batch_idx >= self.train_size:
                self.train_shuffle()
            img_name_class = self._train_image_names_classes[self._train_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                # atk
                if not with_path:
                    # atk
                    path_name = img_name_class[0].split("/")
                    atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
                    atk_res = self._load_image([atk_path+".npy", img_name_class[1]], 
                                               self.onehot, self.normalize, self.biased,
                                               is_np=True)
                    assert atk_res is not None
                    atk_images[idx, :, :, :] = atk_res[0]
                #
                #
                paths.append(img_name_class[0])
                idx += 1
            self._train_batch_idx += 1
        if with_path:
            return images, labels, paths
        else:
            return images, labels, atk_images
    
    def next_valid_batch(self, batch_size, with_path=False):
        images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, 200], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        paths = []
        idx = 0
        while idx < batch_size:
            if self._valid_batch_idx >= self.valid_size:
                self.valid_shuffle()
            img_name_class = self._valid_image_names_classes[self._valid_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                # atk
                if not with_path:
                    # atk
                    path_name = img_name_class[0].split("/")
                    atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
                    atk_res = self._load_image([atk_path+".npy", img_name_class[1]], 
                                               self.onehot, self.normalize, self.biased,
                                               is_np=True)
                    assert atk_res is not None
                    atk_images[idx, :, :, :] = atk_res[0]
                #
                #
                paths.append(img_name_class[0])
                idx += 1
            self._valid_batch_idx += 1
        if with_path:
            return np.array(images), np.array(labels), paths
        else:
            return np.array(images), np.array(labels), atk_images
    
    def next_test_batch(self, batch_size, with_path=False):
        images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        if self.onehot:
            labels = np.zeros([batch_size, 200], dtype='float32')
        else:
            labels = np.zeros([batch_size, 1], dtype='float32')
        if not with_path:
            atk_images = np.ndarray([batch_size, 64, 64, 3], dtype='float32')
        idx = 0
        paths = []
        while idx < batch_size:
            if self._test_batch_idx >= self.test_size:
                self.test_shuffle()
            img_name_class = self._test_image_names_classes[self._test_batch_idx]
            res = self._load_image(img_name_class, self.onehot, self.normalize, self.biased)
            if res is not None:
                images[idx, :, :, :] = res[0]
                labels[idx, :] = res[1]
                if not with_path:
                    # atk
                    path_name = img_name_class[0].split("/")
                    atk_path = "/".join(path_name[:-1] + [self.adv_path_prefix, path_name[-1]])
                    atk_res = self._load_image([atk_path+".npy", img_name_class[1]], 
                                               self.onehot, self.normalize, self.biased,
                                               is_np=True)
                    assert atk_res is not None
                    atk_images[idx, :, :, :] = atk_res[0]
                #
                paths.append(img_name_class[0])
                idx += 1
            self._test_batch_idx += 1
        if with_path:
            return np.array(images), np.array(labels), paths
        else:
            return np.array(images), np.array(labels), atk_images

