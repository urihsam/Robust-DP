""" Residual Encoder
"""
from dependency import *
import utils.net_element as ne
from nn.ae.resenc import RESENC
from utils.decorator import lazy_method, lazy_method_no_scope


class VRESENC(RESENC):
    def __init__(self, 
                 # conv layers
                 conv_filter_sizes=[4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[32, 64, 128], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4],
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[[3,3], [1,1]], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=[0.2, 0.2],
                 # out conv layer
                 out_filter_size=[3,3],
                 out_stride = [1,1],
                 out_padding = "SAME", 
                 out_channel_size=3,
                 out_leaky_ratio=0.2,
                 # img channel
                 img_channel=None,
                 # switch
                 out_norm = None,
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
        # res layers
        self.num_res_block = num_res_block
        self.res_block_size = res_block_size
        if isinstance(res_filter_sizes[0], list):
            self.res_filter_sizes = res_filter_sizes
        else:
            self.res_filter_sizes = [res_filter_sizes] * self.res_block_size
        if isinstance(res_strides[0], list):
            self.res_strides = res_strides
        else:
            self.res_strides = [res_strides] * self.res_block_size
        if isinstance(res_padding, str):
            self.res_padding = [res_padding] * self.res_block_size
        else:
            self.res_padding = res_padding
        if isinstance(res_leaky_ratio, list):
            self.res_leaky_ratio = res_leaky_ratio
        else:
            self.res_leaky_ratio = [res_leaky_ratio] * self.res_block_size
        # out conv layer
        self.out_filter_size = out_filter_size
        self.out_stride = out_stride
        self.out_padding = out_padding
        self.out_channel_size = out_channel_size
        self.out_leaky_ratio = out_leaky_ratio
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.out_norm = out_norm
        self.use_norm = use_norm


        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.res_filters, self.res_biases = self.res_weights_biases()
        self.out_mu_filter, self.out_mu_bias, self.out_std_filter, self.out_std_bias = self.out_weight_bias()


    @lazy_method
    def out_weight_bias(self):
        out_mu_filter, out_mu_bias, _ = self._conv_weights_biases("W_out_mu_", "b_out_mu_", [self.out_filter_size], self.conv_channel_sizes[-1], [self.out_channel_size])
        out_std_filter, out_std_bias, _ = self._conv_weights_biases("W_out_std_", "b_out_std_", [self.out_filter_size], self.conv_channel_sizes[-1], [self.out_channel_size])
        return out_mu_filter, out_mu_bias, out_std_filter, out_std_bias
    
    @lazy_method_no_scope
    def out_layer_(self, inputs, out_name, curr_filter, curr_bias, out_norm):
        net = inputs
        # convolution
        net = ne.conv2d(net, filters=curr_filter, biases=curr_bias, 
                        strides=self.out_stride, 
                        padding=self.out_padding)
        if out_norm == "BATCH":
            net = ne.batch_norm(net, self.is_training)
        elif out_norm == "LAYER":
            net = ne.layer_norm(net, self.is_training)
        #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
        net = ne.leaky_relu(net, self.out_leaky_ratio)
        #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name=out_name)
        #import pdb; pdb.set_trace()
        return net

    @lazy_method
    def out_layer(self, inputs, W_name="W_out_", b_name="b_out_"):
        
        filter_name = "{}mu_{}".format(W_name, 0)
        bias_name = "{}mu_{}".format(b_name, 0)
        curr_filter = self.out_mu_filter[filter_name]
        curr_bias = self.out_mu_bias[bias_name]
        mu = self.out_layer_(inputs, "Out_mu", curr_filter, curr_bias, self.out_norm)
        
        
        filter_name = "{}std_{}".format(W_name, 0)
        bias_name = "{}std_{}".format(b_name, 0)
        curr_filter = self.out_std_filter[filter_name]
        curr_bias = self.out_std_bias[bias_name]
        log_std = self.out_layer_(inputs, "Out_std", curr_filter, curr_bias, "NONE")
        log_std = tf.minimum(log_std, 5.5) # upper bound
        log_std = tf.maximum(log_std, 4.5) # lower bound
        std = tf.exp(log_std)
        
        dist = tf.distributions.Normal(mu, std)
        return dist


    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        conv_res = self.conv_res_groups(data)
        #assert res.get_shape().as_list()[1:] == self.res_out_shape
        out = self.out_layer(conv_res)
        return out


    def tf_load(self, sess, path, scope, name='deep_vresenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_vresenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
