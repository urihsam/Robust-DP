""" Convolutional Autoconvder
"""
from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope


class CAE:
    def __init__(self, 
                 output_low_bound, 
                 output_up_bound,
                 # relu bounds
                 nonlinear_low_bound,
                 nonlinear_up_bound,
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[128, 128, 128, 64, 64, 64, 3], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.1],
                 # deconv layers
                 decv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[3, 64, 64, 64, 128, 128, 128],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.01],
                 # encoder fc layers
                 enfc_state_sizes=[4096], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # bottleneck
                 central_state_size=2048, 
                 # decoder fc layers
                 defc_state_sizes=[4096],
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0.75, 0],
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = None
                ):
        self.output_low_bound = output_low_bound
        self.output_up_bound = output_up_bound
        # relu bounds
        self.nonlinear_low_bound = nonlinear_low_bound
        self.nonlinear_up_bound = nonlinear_up_bound
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
        # encoder fc layers
        self.enfc_state_sizes = enfc_state_sizes 
        self.enfc_leaky_ratio = enfc_leaky_ratio
        self.enfc_drop_rate = enfc_drop_rate
        # bottleneck
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

        self.conv_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.conv_channel_sizes[-1]]
        self.decv_in_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.decv_channel_sizes[0]]
        self.decv_out_dim = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS]

        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.enfc_weights, self.enfc_biases, self.num_enfc = self.enfc_weights_biases()
        self.defc_weights, self.defc_biases, self.num_defc = self.defc_weights_biases()
        self.decv_filters, self.decv_biases, self.num_decv = self.decv_weights_biases()


    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv", "b_conv", self.conv_filter_sizes, self.img_channel, self.conv_channel_sizes)


    @lazy_method
    def decv_weights_biases(self):
        in_channel = self.decv_channel_sizes[0]
        channel_sizes = self.decv_channel_sizes[1:] + [self.img_channel]
        return self._conv_weights_biases("W_decv", "b_decv", self.decv_filter_sizes, in_channel, channel_sizes, True)
    

    def _conv_weights_biases(self, W_name, b_name, filter_sizes, in_channel, channel_sizes, transpose=False):
        num_layer = len(channel_sizes)
        _weights = {}
        _biases = {}
        for idx in range(num_layer):
            W_key = "{}{}".format(W_name, idx)
            if transpose:
                W_shape = filter_sizes[idx]+[channel_sizes[idx], in_channel]
            else:
                W_shape = filter_sizes[idx]+[in_channel, channel_sizes[idx]]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}".format(b_name, idx)
            b_shape = [channel_sizes[idx]]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_channel = channel_sizes[idx]

            # tensorboard
            tf.summary.histogram("Filter_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])

        return _weights, _biases, num_layer


    @lazy_method
    def enfc_weights_biases(self):
        in_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        state_sizes = self.enfc_state_sizes + [self.central_state_size]
        return self._fc_weights_biases("W_enfc", "b_enfc", in_size, state_sizes)

    
    @lazy_method
    def defc_weights_biases(self):
        in_size = self.central_state_size
        out_size = self.decv_in_shape[0] * self.decv_in_shape[1] * self.decv_in_shape[2]
        state_sizes = self.defc_state_sizes + [out_size]
        return self._fc_weights_biases("W_defc", "b_defc", in_size, state_sizes)

    
    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        for idx in range(num_layer):
            W_key = "{}{}".format(W_name, idx)
            W_shape = [in_size, state_sizes[idx]]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}".format(b_name, idx)
            b_shape = [state_sizes[idx]]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = state_sizes[idx]

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])

        return _weights, _biases, num_layer

    
    @lazy_method
    def conv_layers(self, inputs, W_name="W_conv", b_name="b_conv"):
        net = tf.reshape(inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
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
        #import pdb; pdb.set_trace()
        return net

    
    @lazy_method
    def decv_layers(self, inputs, W_name="W_decv", b_name="b_decv"):
        net = inputs
        for layer_id in range(self.num_decv):
            filter_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_filter = self.decv_filters[filter_name]
            curr_bias = self.decv_biases[bias_name]
            # de-convolution
            net = ne.conv2d_transpose(net, out_dim=self.decv_out_dim, 
                                      filters=curr_filter, biases=curr_bias,
                                      strides=self.decv_strides[layer_id], 
                                      padding=self.decv_padding[layer_id])
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            if layer_id == self.num_decv-1: # last layer
                net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.nonlinear_low_bound, self.nonlinear_up_bound) # Nonlinear act
                #net = tf.tanh(net)
            else:
                #net = ne.leaky_brelu(net, self.decv_leaky_ratio[layer_id], self.output_low_bound, self.output_up_bound) # Nonlinear act
                net = ne.leaky_relu(net, self.decv_leaky_ratio[layer_id])
                #net = ne.elu(net)
        net = ne.brelu(net, self.output_low_bound, self.output_up_bound) # clipping the final result 
        net = tf.identity(net, name='output')
        net = tf.reshape(net, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        #import pdb; pdb.set_trace()
        return net

    
    @lazy_method
    def encoder(self, inputs):
        conv = self.conv_layers(inputs)
        assert conv.get_shape().as_list()[1:] == self.conv_out_shape
        enfc = self.enfc_layers(conv)
        assert enfc.get_shape().as_list()[1:] == [self.central_state_size]
        return enfc
    

    @lazy_method
    def decoder(self, inputs):
        defc = self.defc_layers(inputs)
        assert defc.get_shape().as_list()[1:] == self.decv_in_shape
        generated = self.decv_layers(defc)
        assert generated.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel]
        return generated


    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        states = self.encoder(data)
        generated = self.decoder(states)
        if FLAGS.NORMALIZE:
            out = ne.brelu(generated)
        else:
            out = ne.brelu(generated, low_bound=0.0, up_bound=255.0)
        return out

    def tf_load(self, sess, path, name='deep_cae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, name='deep_cae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.save(sess, path+'/'+name+spec)
