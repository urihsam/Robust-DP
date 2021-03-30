""" Convolutional Encoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class CENC(ABCCNN):
    def __init__(self, 
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[128, 256, 512, 512, 256, 128, 3], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                 # encoder fc layers
                 use_fc = True,
                 enfc_state_sizes=[1024], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # bottleneck
                 central_state_size=2048, 
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
        # encoder fc layers
        self.use_fc = use_fc
        if self.use_fc:
            self.enfc_state_sizes = enfc_state_sizes 
            self.enfc_leaky_ratio = enfc_leaky_ratio
            self.enfc_drop_rate = enfc_drop_rate
            # bottleneck
            self.central_state_size = central_state_size
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.use_norm = use_norm

        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        if self.use_fc:
            self.enfc_weights, self.enfc_biases, self.num_enfc = self.enfc_weights_biases()
        


    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv", "b_conv", self.conv_filter_sizes, self.img_channel, self.conv_channel_sizes)


    @lazy_method
    def enfc_weights_biases(self):
        in_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        state_sizes = self.enfc_state_sizes + [self.central_state_size]
        return self._fc_weights_biases("W_enfc", "b_enfc", in_size, state_sizes)
    

    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes, sampling=False):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        def _func(in_size, out_size, idx, postfix=""):
            W_key = "{}{}{}".format(W_name, idx, postfix)
            W_shape = [in_size, out_size]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}{}".format(b_name, idx, postfix)
            b_shape = [out_size]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = out_size

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])
            
            return in_size
        
        for idx in range(num_layer-1):
            in_size = _func(in_size, state_sizes[idx], idx)
        # Last layer
        if sampling:
            if self.vtype == "gauss":
                for postfix in ["_mu", "_sigma"]:
                    _func(in_size, state_sizes[num_layer-1], num_layer-1, postfix)
            elif self.vtype == "vmf":
                _func(in_size, state_sizes[num_layer-1], num_layer-1, "_mu")
                _func(in_size, 1, num_layer-1, "_sigma")
            else:
                raise NotImplementedError("vtype must be 'gauss' or 'vmf'")
        else:
             _func(in_size, state_sizes[num_layer-1], num_layer-1)
        #import pdb; pdb.set_trace()

        return _weights, _biases, num_layer

    
    @lazy_method
    def conv_layers(self, inputs, W_name="W_conv", b_name="b_conv"):
        net = inputs
        for layer_id in range(self.num_conv):
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_filter = self.conv_filters[filter_name]
            curr_bias = self.conv_biases[bias_name]
            # convolution
            net = ne.conv2d(net, filters=curr_filter, biases=curr_bias, 
                            strides=self.conv_strides[layer_id], 
                            padding=self.conv_padding[layer_id])
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.conv_leaky_ratio[layer_id])
            #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name='output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def enfc_layers(self, inputs, W_name="W_enfc", b_name="b_enfc"):
        net = tf.reshape(inputs, [-1, self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]])
        for layer_id in range(self.num_enfc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.enfc_weights[weight_name]
            curr_bias = self.enfc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training, axis=1)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.enfc_leaky_ratio[layer_id], self.enfc_low_bound[layer_id], self.enfc_up_bound[layer_id]) # Nonlinear act
            net = ne.leaky_relu(net, self.enfc_leaky_ratio[layer_id])
            net = ne.drop_out(net, self.enfc_drop_rate[layer_id], self.is_training)
            #net = ne.elu(net)

        net = tf.identity(net, name='output')
        return net


    @lazy_method
    def evaluate(self, inputs, is_training):
        self.is_training = is_training
        conv = self.conv_layers(inputs)
        
        if self.use_fc:
            enfc = self.enfc_layers(conv)
            assert enfc.get_shape().as_list()[1:] == [self.central_state_size]
            return enfc
        else: return conv

    def tf_load(self, sess, path, scope, name='deep_cenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_cenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
