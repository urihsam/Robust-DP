""" Residual Decoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class RESDEC(ABCCNN):
    def __init__(self,
                 output_low_bound,
                 output_up_bound,
                 # deconv layers
                 decv_filter_sizes=[4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[128, 64, 32],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.4, 0.4, 0.4],
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[[3,3], [1,1]], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=[0.2, 0.2],
                 # in fc layer
                 in_state=768,
                 in_fc_states=[768],
                 in_leaky_ratio=0.2,
                 # img channel
                 img_channel=None,
                 # switch
                 in_norm = None,
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
        # in conv layer
        self.in_state = in_state
        if isinstance(in_fc_states, list):
            self.in_fc_states = in_fc_states
        else:
            self.in_fc_states = [in_fc_states]
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

        self.in_weight, self.in_bias = self.in_weight_bias()
        self.res_filters, self.res_biases = self.res_weights_biases()
        self.decv_filters, self.decv_biases, self.num_decv = self.decv_weights_biases()


    @lazy_method
    def decv_weights_biases(self):
        in_channel = self.decv_channel_sizes[0]
        channel_sizes = self.decv_channel_sizes[1:] + [self.img_channel]
        return self._conv_weights_biases("W_decv_", "b_decv_", self.decv_filter_sizes, in_channel, channel_sizes, True)

    
    @lazy_method
    def res_weights_biases(self):
        Ws = {}; bs = {}
        for g_idx in range(len(self.decv_channel_sizes)):
            n_channel = self.decv_channel_sizes[g_idx]
            res_channels = [n_channel] * self.res_block_size
            for idx in range(self.num_res_block):
                W_name = "W_g{}_res{}_".format(g_idx, idx)
                b_name = "b_g{}_res{}_".format(g_idx, idx)
                W_, b_, _ = self._conv_weights_biases(W_name, b_name, self.res_filter_sizes, n_channel, res_channels, True)
                Ws.update(W_)
                bs.update(b_)
        return Ws, bs


    @lazy_method
    def in_weight_bias(self):
        in_size = self.in_state
        state_size = self.in_fc_states
        W, b, _ = self._fc_weights_biases("W_in_", "b_in_", in_size, state_size, init_type="XV_1")
        return W, b


    @lazy_method
    def in_layer(self, inputs, W_name="W_in_", b_name="b_in_"):
        net = inputs
        # fc
        h, w, c = net.get_shape().as_list()[1:]
        assert h*w*c == self.in_state
        net = tf.reshape(net, [-1, self.in_state])

        for layer_id in range(len(self.in_fc_states)):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.in_weight[weight_name]
            curr_bias = self.in_bias[bias_name]

            # batch normalization
            if self.in_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.in_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.in_leaky_ratio)
            net = ne.fully_conn(net, curr_weight, curr_bias)
        
        out_channel = self.in_fc_states[-1]//h//w
        assert h*w*out_channel == self.in_fc_states[-1]
        net = tf.reshape(net, [-1, h, w, out_channel])
        net = tf.identity(net, name='in_output')
        #import pdb; pdb.set_trace()
        return net

    
    @lazy_method
    def res_blocks(self, inputs, W_name, b_name):
        net = inputs
        for res_id in range(self.num_res_block):
            res_net = net
            for layer_id in range(self.res_block_size):
                filter_name = "{}{}_{}".format(W_name, res_id, layer_id)
                bias_name = "{}{}_{}".format(b_name, res_id, layer_id)
                curr_filter = self.res_filters[filter_name]
                curr_bias = self.res_biases[bias_name]
                
                #net = ne.leaky_brelu(net, self.res_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.res_leaky_ratio[layer_id])
                # convolution
                net = ne.conv2d_transpose(net, 
                                        filters=curr_filter, biases=curr_bias, 
                                        strides=self.res_strides[layer_id], 
                                        padding=self.res_padding[layer_id])
                
                
            net += res_net
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
        net = tf.identity(net, name='res_output')
        #import pdb; pdb.set_trace()
        return net

    
    @lazy_method
    def decv_res_groups(self, inputs, W_name="W_decv_", b_name="b_decv_"):
        net = inputs
        for layer_id in range(self.num_decv):
            #res blocks
            W_res_name = "W_g{}_res".format(layer_id)
            b_res_name = "b_g{}_res".format(layer_id)
            net = self.res_blocks(net, W_res_name, b_res_name)
            # decv
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_filter = self.decv_filters[filter_name]
            curr_bias = self.decv_biases[bias_name]
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
            # de-convolution
            net = ne.conv2d_transpose(net,
                                      filters=curr_filter, biases=curr_bias,
                                      strides=self.decv_strides[layer_id], 
                                      padding=self.decv_padding[layer_id])
            

        net = ne.brelu(net, self.output_low_bound, self.output_up_bound) # clipping the final result 
        net = tf.identity(net, name='output')
        net = tf.reshape(net, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def evaluate(self, inputs, is_training):
        self.is_training = is_training
        #assert inputs.get_shape().as_list()[1:] == self.res_in_shape
        in_ = self.in_layer(inputs)
        generated = self.decv_res_groups(in_)
        #assert generated.get_shape().as_list()[1:] == self.decv_out_shape
        return generated

    def tf_load(self, sess, path, scope, name='deep_resdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_resdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
