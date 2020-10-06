""" Residual Encoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class ATTRESENC(ABCCNN):
    def __init__(self,
                 # attention layer
                 attention_type,
                 att_pos_idx=1,
                 att_f_filter_size = [1,1],
                 att_f_strides = [1,1],
                 att_f_padding = "SAME",
                 att_f_channel_size = 16,
                 #
                 att_g_filter_size = [1,1],
                 att_g_strides = [1,1],
                 att_g_padding = "SAME",
                 att_g_channel_size = 16,
                 #
                 att_h_filter_size=[1,1],
                 att_h_strides = [1,1],
                 att_h_padding = "SAME",
                 att_h_channel_size = 16,
                 #
                 att_o_filter_size=[1,1],
                 att_o_strides = [1,1],
                 att_o_padding = "SAME",
                 # conv layers
                 conv_filter_sizes = [4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]],
                 conv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes = [32, 64, 128], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio = [0.4, 0.4, 0.4],
                 conv_drop_rate=[0.4, 0.4, 0.4],
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[[3,3], [1,1]], #[[3,3], [3,3], [3,3], [3,3], [3,3]],
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=[0.2, 0.2],
                 res_drop_rate=[0.4, 0.4],
                 # out conv layer
                 use_out_layer = False,
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
        # attention
        self.attention_type = attention_type
        self.att_pos_idx = len(conv_filter_sizes) + att_pos_idx if att_pos_idx < 0 else att_pos_idx
        self.att_f_filter_size = att_f_filter_size
        self.att_f_strides = att_f_strides
        self.att_f_padding = att_f_padding
        self.att_f_channel_size = att_f_channel_size
        #
        self.att_g_filter_size = att_g_filter_size
        self.att_g_strides = att_g_strides
        self.att_g_padding = att_g_padding
        self.att_g_channel_size = att_g_channel_size
        #
        self.att_h_filter_size = att_h_filter_size
        self.att_h_strides = att_h_strides
        self.att_h_padding = att_h_padding
        self.att_h_channel_size = att_h_channel_size
        #
        self.att_o_filter_size = att_o_filter_size
        self.att_o_strides = att_o_strides
        self.att_o_padding = att_o_padding
        self.att_o_channel_size = att_h_channel_size
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
        self.conv_drop_rate = conv_drop_rate
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
        if isinstance(res_drop_rate, list):
            self.res_drop_rate = res_drop_rate
        else:
            self.res_drop_rate = [res_drop_rate] * self.res_block_size
        self.use_out_layer =  use_out_layer
        if self.use_out_layer:
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

        self.att_filters, self.att_gamma = self.att_weights()
        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.res_filters, self.res_biases = self.res_weights_biases()
        if self.use_out_layer:
            self.out_filter, self.out_bias, _ = self.out_weight_bias()


    @lazy_method
    def att_weights(self):
        W = {}
        f_filters, _, _ = self._conv_weights_biases("W_att_f_", None,
                                                    [self.att_f_filter_size],
                                                    self.conv_channel_sizes[self.att_pos_idx],
                                                    [self.att_f_channel_size],
                                                    no_bias=True)
        g_filters, _, _ = self._conv_weights_biases("W_att_g_", None,
                                                    [self.att_g_filter_size],
                                                    self.conv_channel_sizes[self.att_pos_idx],
                                                    [self.att_g_channel_size],
                                                    no_bias=True)
        h_filters, _, _ = self._conv_weights_biases("W_att_h_", None,
                                                    [self.att_h_filter_size],
                                                    self.conv_channel_sizes[self.att_pos_idx],
                                                    [self.att_h_channel_size],
                                                    no_bias=True)
        o_filters, _, _ = self._conv_weights_biases("W_att_o_", None,
                                                    [self.att_o_filter_size],
                                                    self.att_o_channel_size,
                                                    [self.conv_channel_sizes[self.att_pos_idx]],
                                                    no_bias=True)
        W.update(f_filters); W.update(g_filters); W.update(h_filters); W.update(o_filters)

        gamma = tf.get_variable("Gamma_att", [1], initializer=tf.constant_initializer(0.0))
        return W, gamma


    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv_", "b_conv_", self.conv_filter_sizes, self.img_channel, self.conv_channel_sizes)


    @lazy_method
    def res_weights_biases(self):
        Ws = {}; bs = {}
        for g_idx in range(len(self.conv_channel_sizes)):
            n_channel = self.conv_channel_sizes[g_idx]
            res_channels = [n_channel] * self.res_block_size
            for idx in range(self.num_res_block):
                W_name = "W_g{}_res{}_".format(g_idx, idx)
                b_name = "b_g{}_res{}_".format(g_idx, idx)
                W_, b_, _ = self._conv_weights_biases(W_name, b_name, self.res_filter_sizes, n_channel, res_channels)
                Ws.update(W_)
                bs.update(b_)
        return Ws, bs


    @lazy_method
    def out_weight_bias(self):
        return self._conv_weights_biases("W_out_", "b_out_", [self.out_filter_size], self.conv_channel_sizes[-1], [self.out_channel_size])

    @lazy_method
    def conv_res_groups(self, inputs, W_name="W_conv_", b_name="b_conv_"):
        #net = tf.reshape(inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        def _form_groups(net, start_layer, end_layer):
            for layer_id in range(start_layer, end_layer):
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
                net = ne.drop_out(net, self.conv_drop_rate[layer_id], self.is_training)

                # res blocks
                W_res_name = "W_g{}_res".format(layer_id)
                b_res_name = "b_g{}_res".format(layer_id)
                net = self.res_blocks(net, W_res_name, b_res_name, scope="RES_{}".format(layer_id))
            return net
        net = inputs
        net = _form_groups(net, 0, self.att_pos_idx+1)
        # attention
        net = self.att_layer(net)
        net = _form_groups(net, self.att_pos_idx+1, self.num_conv)
        net = tf.identity(net, name='conv_output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method_no_scope
    def res_blocks(self, inputs, W_name, b_name, scope):
        with tf.variable_scope(scope):
            net = inputs
            for res_id in range(self.num_res_block):
                res_net = net
                for layer_id in range(self.res_block_size):
                    filter_name = "{}{}_{}".format(W_name, res_id, layer_id)
                    bias_name = "{}{}_{}".format(b_name, res_id, layer_id)
                    curr_filter = self.res_filters[filter_name]
                    curr_bias = self.res_biases[bias_name]

                    # convolution
                    net = ne.conv2d(net, filters=curr_filter, biases=curr_bias,
                                    strides=self.res_strides[layer_id],
                                    padding=self.res_padding[layer_id])
                    if self.use_norm == "BATCH":
                        net = ne.batch_norm(net, self.is_training)
                    elif self.use_norm == "LAYER":
                        net = ne.layer_norm(net, self.is_training)

                    #net = ne.leaky_brelu(net, self.res_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
                    net = ne.leaky_relu(net, self.res_leaky_ratio[layer_id])
                    net = ne.drop_out(net, self.res_drop_rate[layer_id], self.is_training)



                net += res_net

            net = tf.identity(net, name='res_output')
            #import pdb; pdb.set_trace()
            return net


    @lazy_method
    def att_layer(self, inputs, W_name="W_att_"):
        net = inputs
        f = ne.conv2d(net, filters=self.att_filters[W_name+"f_0"], biases=None,
                      strides=self.att_f_strides, padding=self.att_f_padding) # [b, h, w, c]
        if self.attention_type == "GOOGLE":
            f = ne.max_pool_2x2(f)
        g = ne.conv2d(net, filters=self.att_filters[W_name+"g_0"], biases=None,
                      strides=self.att_g_strides, padding=self.att_g_padding) # [b, h, w, c]
        h = ne.conv2d(net, filters=self.att_filters[W_name+"h_0"], biases=None,
                      strides=self.att_h_strides, padding=self.att_h_padding) # [b, h, w, c]
        if self.attention_type == "GOOGLE":
            h = ne.max_pool_2x2(h)

        # N = h * w
        s = tf.matmul(ne.hw_flatten(g), ne.hw_flatten(f), transpose_b=True) # [b, N, N]
        beta = ne.softmax(s)  # attention map, [b, N, N]
        o = tf.matmul(beta, ne.hw_flatten(h)) # [b, N, C]
        o = tf.reshape(o, shape=[tf.shape(inputs)[0]] + inputs.get_shape().as_list()[1:-1]+[self.att_o_channel_size]) # [b, h, w, C]
        o = ne.conv2d(o, filters=self.att_filters[W_name+"o_0"], biases=None,
                      strides=self.att_o_strides, padding=self.att_o_padding) # [b, h, w, c]
        net = self.att_gamma * o + net

        return net


    @lazy_method
    def out_layer(self, inputs, W_name="W_out_", b_name="b_out_"):
        layer_id = 0
        net = inputs
        filter_name = "{}{}".format(W_name, layer_id)
        bias_name = "{}{}".format(b_name, layer_id)
        curr_filter = self.out_filter[filter_name]
        curr_bias = self.out_bias[bias_name]
        # convolution
        net = ne.conv2d(net, filters=curr_filter, biases=curr_bias,
                        strides=self.out_stride,
                        padding=self.out_padding)
        if self.out_norm == "BATCH":
            net = ne.batch_norm(net, self.is_training)
        elif self.out_norm == "LAYER":
            net = ne.layer_norm(net, self.is_training)
        #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
        net = ne.leaky_relu(net, self.out_leaky_ratio)
        #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name='out_output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        conv_res = self.conv_res_groups(data)
        #assert res.get_shape().as_list()[1:] == self.res_out_shape
        if self.use_out_layer:
            conv_res = self.out_layer(conv_res)
        return conv_res


    def tf_load(self, sess, path, scope, name='deep_resenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_resenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
