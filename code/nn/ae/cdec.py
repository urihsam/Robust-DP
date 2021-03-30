""" Convolutional Decoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class CDEC(ABCCNN):
    def __init__(self,
                 output_low_bound,
                 output_up_bound,
                 # deconv layers
                 decv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[3, 128, 256, 512, 512, 256, 128],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                 # bottleneck
                 use_fc = True,
                 central_state_size=2048, 
                 # decoder fc layers
                 defc_state_sizes=[1024],
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0.75, 0],
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = None
                ):
        self.output_low_bound = output_low_bound
        self.output_up_bound = output_up_bound
        # deconv layers
        if isinstance(decv_filter_sizes[0], list):
            self.decv_filter_sizes = decv_filter_sizes
        else:
            self.decv_filter_sizes = [decv_filter_sizes] * len(decv_channel_sizes)
        if isinstance(decv_strides[0], list):
            self.decv_strides = decv_strides
        else:
            self.decv_strides = [decv_strides] * len(decv_channel_sizes)
        if isinstance(decv_padding, str):
            self.decv_padding = [decv_padding] * len(decv_channel_sizes)
        else:
            self.decv_padding = decv_padding
        self.decv_channel_sizes = decv_channel_sizes
        self.decv_leaky_ratio = decv_leaky_ratio
        # bottleneck
        self.use_fc = use_fc
        if self.use_fc:
            self.central_state_size = central_state_size
            # decoder fc layers
            self.defc_state_sizes = defc_state_sizes
            self.defc_leaky_ratio = defc_leaky_ratio
            self.defc_drop_rate = defc_drop_rate
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.use_norm = use_norm

        self.decv_in_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.decv_channel_sizes[0]]
        self.decv_out_dim = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS]

        if self.use_fc:
            self.defc_weights, self.defc_biases, self.num_defc = self.defc_weights_biases()
        self.decv_filters, self.decv_biases, self.num_decv = self.decv_weights_biases()


    @lazy_method
    def decv_weights_biases(self):
        in_channel = self.decv_channel_sizes[0]
        channel_sizes = self.decv_channel_sizes[1:] + [self.img_channel]
        return self._conv_weights_biases("W_decv", "b_decv", self.decv_filter_sizes, in_channel, channel_sizes, transpose=True)


    @lazy_method
    def defc_weights_biases(self):
        in_size = self.central_state_size
        out_size = self.decv_in_shape[0] * self.decv_in_shape[1] * self.decv_in_shape[2]
        state_sizes = self.defc_state_sizes + [out_size]
        return self._fc_weights_biases("W_defc", "b_defc", in_size, state_sizes)

    
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
    def defc_layers(self, inputs, W_name="W_defc", b_name="b_defc"):
        net = inputs
        for layer_id in range(self.num_defc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.defc_weights[weight_name]
            curr_bias = self.defc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training, axis=1)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            
            #net = ne.leaky_brelu(net, self.defc_leaky_ratio[layer_id], self.defc_low_bound[layer_id], self.defc_up_bound[layer_id]) # Nonlinear act
            net = ne.leaky_relu(net, self.defc_leaky_ratio[layer_id])
            net = ne.drop_out(net, self.defc_drop_rate[layer_id], self.is_training)
            #net = ne.elu(net)

        net = tf.identity(net, name='output')
        net = tf.reshape(net, [-1] + self.decv_in_shape)
        
        return net

    
    @lazy_method
    def decv_layers(self, inputs, W_name="W_decv", b_name="b_decv"):
        net = inputs
        for layer_id in range(self.num_decv):
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            #import pdb; pdb.set_trace()
            curr_filter = self.decv_filters[filter_name]
            curr_bias = self.decv_biases[bias_name]
            # de-convolution
            net = ne.conv2d_transpose(net,
                                      filters=curr_filter, biases=curr_bias,
                                      strides=self.decv_strides[layer_id], 
                                      padding=self.decv_padding[layer_id])
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            if layer_id == self.num_decv-1: # last layer
                #net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.nonlinear_low_bound, self.nonlinear_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.decv_leaky_ratio[layer_id])
            else:
                #net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.output_low_bound, self.output_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.decv_leaky_ratio[layer_id])
                #net = ne.elu(net)
        net = ne.brelu(net, self.output_low_bound, self.output_up_bound) # clipping the final result 
        net = tf.identity(net, name='output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def evaluate(self, inputs, is_training):
        self.is_training = is_training
        if self.use_fc:
            defc = self.defc_layers(inputs)
            assert defc.get_shape().as_list()[1:] == self.decv_in_shape
            generated = self.decv_layers(defc)
        else:
            generated = self.decv_layers(inputs)
        return generated

    def tf_load(self, sess, path, scope, name='deep_cdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_cdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
