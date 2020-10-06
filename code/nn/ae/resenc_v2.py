""" Residual Encoder
"""
from dependency import *
import utils.net_element as ne
from nn.cnn.abccnn import ABCCNN
from utils.decorator import lazy_method, lazy_method_no_scope


class RESENC(ABCCNN):
    def __init__(self, 
                 # conv layers
                 conv_filter_sizes=[4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[32, 64, 128], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4],
                 conv_drop_rate=[0.4, 0.4, 0.4],
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[[3,3], [1,1]], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=[0.2, 0.2],
                 res_drop_rate=[0.4, 0.4],
                 # out fc layer
                 out_state=768,
                 out_fc_states=[768],
                 out_leaky_ratio=0.2,
                 # img channel
                 img_channel=None,
                 # switch
                 out_norm = None,
                 use_norm = None,
                 use_class_label = False
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
        self.conv_drop_rate = conv_drop_rate
        # res layers
        self.num_res_block = num_res_block
        if self.num_res_block != 0:
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
        # out fc layer
        self.out_state = out_state
        if isinstance(out_fc_states, list):
            self.out_fc_states = out_fc_states
        else:
            self.out_fc_states = [out_fc_states]
        self.out_leaky_ratio = out_leaky_ratio
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.out_norm = out_norm
        self.use_norm = use_norm
        self.use_class_label = use_class_label


        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        if self.num_res_block != 0:
            self.res_filters, self.res_biases = self.res_weights_biases()
        self.out_weight, self.out_bias = self.out_weight_bias()


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
        in_size = self.out_state
        if self.use_class_label:
            in_size += FLAGS.LBL_STATES_SIZE
        state_size = self.out_fc_states
        W, b, _ = self._fc_weights_biases("W_out_", "b_out_", in_size, state_size, init_type="XV_1")
        return W, b

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
                elif self.use_norm == "INSTA":
                    net = ne.instance_norm(net, self.is_training)
                #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.conv_leaky_ratio[layer_id])
                net = ne.drop_out(net, self.conv_drop_rate[layer_id], self.is_training)

                # res blocks
                if self.num_res_block != 0:
                    W_res_name = "W_g{}_res".format(layer_id)
                    b_res_name = "b_g{}_res".format(layer_id)
                    net = self.res_blocks(net, W_res_name, b_res_name, scope="RES_{}".format(layer_id))
            return net
        net = inputs
        
        net = _form_groups(net, 0, self.num_conv)
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
                    elif self.use_norm == "INSTA":
                        net = ne.instance_norm(net, self.is_training)

                    #net = ne.leaky_brelu(net, self.res_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
                    net = ne.leaky_relu(net, self.res_leaky_ratio[layer_id])
                    net = ne.drop_out(net, self.res_drop_rate[layer_id], self.is_training)



                net += res_net

            net = tf.identity(net, name='res_output')
            #import pdb; pdb.set_trace()
            return net
    

    @lazy_method
    def out_layer(self, inputs, label=None, W_name="W_out_", b_name="b_out_"):
        net = inputs
        h, w, c = net.get_shape().as_list()[1:]
        assert h*w*c == self.out_state
        net = tf.reshape(net, [-1, self.out_state])
        if self.use_class_label:
            net = tf.concat([net, label], -1)
        
        for layer_id in range(len(self.out_fc_states)):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.out_weight[weight_name]
            curr_bias = self.out_bias[bias_name]

            net = ne.fully_conn(net, curr_weight, curr_bias)
            if self.out_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.out_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            net = ne.leaky_relu(net, self.out_leaky_ratio)

        out_channel_size = self.out_fc_states[-1]//h//w
        assert h*w*out_channel_size == self.out_fc_states[-1]
        net = tf.reshape(net, [-1, h, w, out_channel_size])
        #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name='out_output')
        #import pdb; pdb.set_trace()
        return net


    @lazy_method
    def evaluate(self, data, is_training, label=None):
        self.is_training = is_training
        conv_res = self.conv_res_groups(data)
        #assert res.get_shape().as_list()[1:] == self.res_out_shape
        out = self.out_layer(conv_res, label)
        return out


    def tf_load(self, sess, path, scope, name='deep_resenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_resenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
