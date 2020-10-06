""" Residual Decoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class ATTRESDEC(ABCCNN):
    def __init__(self,
                 output_low_bound,
                 output_up_bound,
                 # attention layer
                 attention_type,
                 att_pos_idx=-2, # att will be applied after att_pos_idx of decv_res_block
                 att_f_filter_size=[1,1],
                 att_f_strides = [1,1],
                 att_f_padding = "SAME",
                 att_f_channel_size = 16,
                 att_g_filter_size=[1,1],
                 att_g_strides = [1,1],
                 att_g_padding = "SAME",
                 att_g_channel_size = 16,
                 att_h_filter_size=[1,1],
                 att_h_strides = [1,1],
                 att_h_padding = "SAME",
                 att_h_channel_size = 16,
                 att_o_filter_size=[1,1],
                 att_o_strides = [1,1],
                 att_o_padding = "SAME",
                 # deconv layers
                 decv_filter_sizes=[4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[128, 64, 32],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.4, 0.4, 0.4],
                 decv_drop_rate=[0.4, 0.4, 0.4],
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[[3,3], [1,1]], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=[0.2, 0.2],
                 res_drop_rate=[0.4, 0.4],
                 # in conv layer
                 use_in_layer = False,
                 in_filter_size=[3,3],
                 in_stride = [1,1],
                 in_padding = "SAME", 
                 in_channel_size=3,
                 in_leaky_ratio=0.2,
                 # img channel
                 img_channel=None,
                 # switch
                 in_norm = None,
                 use_norm = None
                ):
        self.output_low_bound = output_low_bound
        self.output_up_bound = output_up_bound
        # attention
        self.attention_type = attention_type
        self.att_pos_idx = len(decv_channel_sizes) + att_pos_idx if att_pos_idx < 0 else att_pos_idx
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
        self.decv_drop_rate = decv_drop_rate
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
        # in conv layer
        self.use_in_layer = use_in_layer
        if self.use_in_layer:
            self.in_filter_size = in_filter_size
            self.in_stride = in_stride
            self.in_padding = in_padding
            self.in_channel_size = in_channel_size
            self.in_leaky_ratio = in_leaky_ratio
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.in_norm = in_norm
        self.use_norm = use_norm

        #self.res_in_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.res_channel_sizes[0]]
        #self.decv_in_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.decv_channel_sizes[0]]
        #self.decv_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS]
        self.att_filters, self.att_gamma = self.att_weights()
        if self.use_in_layer:
            self.in_filter, self.in_bias, _ = self.in_weight_bias()
        self.res_filters, self.res_biases = self.res_weights_biases()
        self.decv_filters, self.decv_biases, self.num_decv = self.decv_weights_biases()

    
    @lazy_method
    def att_weights(self):
        W = {}
        if self.att_pos_idx < len(self.decv_channel_sizes) - 1:
            decv_channel_size = self.decv_channel_sizes[self.att_pos_idx + 1]
        else:
            decv_channel_size = self.img_channel
        f_filters, _, _ = self._conv_weights_biases("W_att_f_", None, 
                                                    [self.att_f_filter_size], 
                                                    decv_channel_size, 
                                                    [self.att_f_channel_size],
                                                    no_bias=True)
        g_filters, _, _ = self._conv_weights_biases("W_att_g_", None, 
                                                    [self.att_g_filter_size], 
                                                    decv_channel_size, 
                                                    [self.att_g_channel_size], 
                                                    no_bias=True)
        h_filters, _, _ = self._conv_weights_biases("W_att_h_", None, 
                                                    [self.att_h_filter_size], 
                                                    decv_channel_size, 
                                                    [self.att_h_channel_size], 
                                                    no_bias=True)
        o_filters, _, _ = self._conv_weights_biases("W_att_o_", None, 
                                                    [self.att_o_filter_size], 
                                                    self.att_o_channel_size,
                                                    [decv_channel_size], 
                                                    no_bias=True)
        W.update(f_filters); W.update(g_filters); W.update(h_filters); W.update(o_filters)

        gamma = tf.get_variable("Gamma_att", [1], initializer=tf.constant_initializer(0.0))
        return W, gamma


    @lazy_method
    def decv_weights_biases(self):
        in_channel = self.decv_channel_sizes[0]
        channel_sizes = self.decv_channel_sizes[1:] + [self.img_channel]
        return self._conv_weights_biases("W_decv_", "b_decv_", self.decv_filter_sizes, in_channel, channel_sizes, 
                                         transpose=True)

    
    @lazy_method
    def res_weights_biases(self):
        Ws = {}; bs = {}
        for g_idx in range(len(self.decv_channel_sizes)):
            n_channel = self.decv_channel_sizes[g_idx]
            res_channels = [n_channel] * self.res_block_size
            for idx in range(self.num_res_block):
                W_name = "W_g{}_res{}_".format(g_idx, idx)
                b_name = "b_g{}_res{}_".format(g_idx, idx)
                W_, b_, _ = self._conv_weights_biases(W_name, b_name, self.res_filter_sizes, n_channel, res_channels,
                                                      transpose=True)
                Ws.update(W_)
                bs.update(b_)
        return Ws, bs


    @lazy_method
    def in_weight_bias(self):
        return self._conv_weights_biases("W_in_", "b_in_", [self.in_filter_size], self.in_channel_size, [self.decv_channel_sizes[0]],
                                         transpose=True)


    @lazy_method
    def in_layer(self, inputs, W_name="W_in_", b_name="b_in_"):
        layer_id = 0
        net = inputs
        filter_name = "{}{}".format(W_name, layer_id)
        bias_name = "{}{}".format(b_name, layer_id)
        curr_filter = self.in_filter[filter_name]
        curr_bias = self.in_bias[bias_name]
        # batch normalization
        if self.use_norm == "BATCH":
            net = ne.batch_norm(net, self.is_training)
        elif self.use_norm == "LAYER":
            net = ne.layer_norm(net, self.is_training)
        elif self.use_norm == "INSTA":
            net = ne.instance_norm(net, self.is_training)
        #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
        net = ne.leaky_relu(net, self.in_leaky_ratio)
        # convolution
        net = ne.conv2d_transpose(net, 
                                filters=curr_filter, biases=curr_bias, 
                                strides=self.in_stride, 
                                padding=self.in_padding)
        #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name='in_output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def att_layer(self, inputs, W_name="W_att_"):
        net = inputs
        f = ne.conv2d(net, filters=self.att_filters[W_name+"f_0"], biases=None,
                      strides=self.att_f_strides, padding=self.att_f_padding) # [b, h, w, c]
        g = ne.conv2d(net, filters=self.att_filters[W_name+"g_0"], biases=None,
                      strides=self.att_g_strides, padding=self.att_g_padding) # [b, h, w, c]
        h = ne.conv2d(net, filters=self.att_filters[W_name+"h_0"], biases=None,
                      strides=self.att_h_strides, padding=self.att_h_padding) # [b, h, w, c]
        if self.attention_type == "GOOGLE":
            f = ne.max_pool_2x2(f) # [b, h/2, w/2, c]
            h = ne.max_pool_2x2(h) # [b, h/2, w/2, c]
        elif self.attention_type == "DUOCONV":
            f = ne.max_pool_2x2(ne.max_pool_2x2(f)) # [b, h/4, w/4, c]
            h = ne.max_pool_2x2(ne.max_pool_2x2(h)) # [b, h/4, w/4, c]

        # N = h * w
        s = tf.matmul(ne.hw_flatten(g), ne.hw_flatten(f), transpose_b=True) # [b, N, N]
        beta = ne.softmax(s)  # attention map, [b, N, N]
        o = tf.matmul(beta, ne.hw_flatten(h)) # [b, N, C]
        o = tf.reshape(o, shape=[tf.shape(inputs)[0]] + inputs.get_shape().as_list()[1:-1]+[self.att_o_channel_size]) # [b, h, w, C]
        o = ne.conv2d(o, filters=self.att_filters[W_name+"o_0"], biases=None,
                      strides=self.att_o_strides, padding=self.att_o_padding) # [b, h, w, c]
        net = self.att_gamma * o + net

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
                    net = ne.conv2d_transpose(net, 
                                            filters=curr_filter, biases=curr_bias, 
                                            strides=self.res_strides[layer_id], 
                                            padding=self.res_padding[layer_id])
                    
                    if self.use_norm == "BATCH":
                        net = ne.batch_norm(net, self.is_training)
                    elif self.use_norm == "LAYER":
                        net = ne.layer_norm(net, self.is_training)
                    elif self.use_norm == "INSTA":
                        net = ne.instance_norm(net, self.is_training)
                    net = ne.leaky_relu(net, self.res_leaky_ratio[layer_id])
                    #net = ne.leaky_brelu(net, self.res_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
                    net = ne.drop_out(net, self.res_drop_rate[layer_id], self.is_training)
                    
                    
                    
                net += res_net
            net = tf.identity(net, name='res_output')
            #import pdb; pdb.set_trace()
            return net

    
    @lazy_method
    def decv_res_groups(self, inputs, mask_states, W_name="W_decv_", b_name="b_decv_"):
        def _form_groups(net, start_layer, end_layer):
            for layer_id in range(start_layer, end_layer):
                #res blocks
                W_res_name = "W_g{}_res".format(layer_id)
                b_res_name = "b_g{}_res".format(layer_id)
                net = self.res_blocks(net, W_res_name, b_res_name, scope="RES_{}".format(layer_id))
                # decv
                filter_name = "{}{}".format(W_name, layer_id)
                bias_name = "{}{}".format(b_name, layer_id)
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
                elif self.use_norm == "INSTA":
                    net = ne.instance_norm(net, self.is_training)
                
                if layer_id != end_layer-1:
                    net = ne.leaky_relu(net, self.decv_leaky_ratio[layer_id])
                    net = ne.drop_out(net, self.decv_drop_rate[layer_id], self.is_training)
                
                if layer_id == self.num_decv - 2:
                    # mask
                    if FLAGS.USE_LABEL_MASK:
                        w = net.get_shape().as_list()[1]
                        h = net.get_shape().as_list()[2]
                        c = net.get_shape().as_list()[3]
                        net = tf.reshape(net, [-1, w*h, c])
                        net = tf.matmul(net, mask_states)
                        net = tf.reshape(net, [-1, w, h, c]) 
                        if self.use_norm == "BATCH":
                            net = ne.batch_norm(net, self.is_training)
                        elif self.use_norm == "LAYER":
                            net = ne.layer_norm(net, self.is_training)
                        elif self.use_norm == "INSTA":
                            net = ne.instance_norm(net, self.is_training)
                #import pdb; pdb.set_trace()
            return net

        net = inputs
        net = _form_groups(net, 0, self.att_pos_idx + 1)
        # attention
        net = self.att_layer(net)
        
        net = _form_groups(net, self.att_pos_idx + 1, self.num_decv)
        
        net = tf.identity(net, name='output')
        # net = tf.reshape(net, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def evaluate(self, inputs, is_training, mask_states):
        self.is_training = is_training
        #assert inputs.get_shape().as_list()[1:] == self.res_in_shape
        if self.use_in_layer:
            inputs = self.in_layer(inputs)
        generated = self.decv_res_groups(inputs, mask_states)
        #generated = ne.brelu(generated, self.output_low_bound, self.output_up_bound) # clipping the final result 

        # dense, uniform
        
        ratio = 0.8
        maximum = tf.reduce_max(generated) * ratio
        minimum = tf.reduce_min(generated) * ratio
        positive = tf.minimum(tf.maximum(generated, 0.0), maximum) / maximum * self.output_up_bound
        negative = tf.maximum(tf.minimum(generated, 0.0), minimum) / minimum * self.output_low_bound
        generated = negative + positive
        '''
        #
        # sparse, nonuniform
        #import pdb; pdb.set_trace()
        generated_shape = generated.get_shape().as_list()[1:]
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(ne.flatten(generated)), 1))
        generated = tf.divide(ne.flatten(generated), tf.expand_dims(l2_norm, axis=1))
        generated = tf.reshape(generated, [-1]+generated_shape)
        maximum = tf.maximum(tf.reduce_max(generated), -tf.reduce_min(generated))
        generated = generated / maximum * self.output_up_bound
        #generated = tf.minimum(tf.maximum(generated, self.output_low_bound), self.output_up_bound)
        '''

        return generated

    def tf_load(self, sess, path, scope, name='deep_resdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_resdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
