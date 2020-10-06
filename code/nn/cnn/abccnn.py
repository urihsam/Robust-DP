import abc
from abc import ABC
from dependency import *


class ABCCNN(ABC):
    """
    Abstract Class Convolutional Neural Network
    """
    def __init__(self,
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[128, 128, 64, 64, 3], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.2, 0.2, 0.1],
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = None
                ):
        # conv layers
        if isinstance(conv_filter_sizes[0], list):
            self.conv_filter_sizes = conv_filter_sizes
        else:
            self.conv_filter_sizes = [conv_filter_sizes] * len(conv_channel_sizes)
        if isinstance(conv_strides[0], list):
            self.conv_strides = conv_strides
        else:
            self.conv_strides = [conv_strides] * len(conv_channel_sizes)
        if isinstance(conv_padding, str):
            self.conv_padding = [conv_padding] * len(conv_channel_sizes)
        else:
            self.conv_padding = conv_padding
        self.conv_channel_sizes = conv_channel_sizes
        self.conv_leaky_ratio = conv_leaky_ratio
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.use_norm = use_norm


    def _conv_weights_biases(self, W_name, b_name, filter_sizes, in_channel, channel_sizes, name_shift=0, transpose=False, init_type="XV_1", no_bias=False):
        num_layer = len(channel_sizes)
        _weights = {}
        _biases = {}
        for idx in range(num_layer):
            W_key = "{}{}".format(W_name, idx+name_shift)
            if transpose:
                W_shape = filter_sizes[idx]+[channel_sizes[idx], in_channel]
            else:
                W_shape = filter_sizes[idx]+[in_channel, channel_sizes[idx]]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key, init_type=init_type)
            if no_bias == False:
                b_key = "{}{}".format(b_name, idx+name_shift)
                b_shape = [channel_sizes[idx]]
                _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_channel = channel_sizes[idx]

            # tensorboard
            tf.summary.histogram("Filter_"+W_key, _weights[W_key])
            if no_bias == False:
                tf.summary.histogram("Bias_"+b_key, _biases[b_key])

        return _weights, _biases, num_layer


    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes, init_type="HE", sampling=False, no_bias=False):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        def _func(in_size, out_size, idx, postfix=""):
            W_key = "{}{}{}".format(W_name, idx, postfix)
            W_shape = [in_size, out_size]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key, init_type=init_type)

            if no_bias == False:
                b_key = "{}{}{}".format(b_name, idx, postfix)
                b_shape = [out_size]
                _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = out_size

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            if no_bias == False:
                tf.summary.histogram("Bias_"+b_key, _biases[b_key])
            
            return in_size
        
        for idx in range(num_layer):
            in_size = _func(in_size, state_sizes[idx], idx)

        return _weights, _biases, num_layer


    @abc.abstractmethod
    def evaluate(self, data, is_training):
        pass

    @abc.abstractmethod
    def tf_load(self, sess, path, name='cnn.ckpt', spec=""):
        pass

    @abc.abstractmethod
    def tf_save(self, sess, path, name='cnn.ckpt', spec=""):
        pass

        