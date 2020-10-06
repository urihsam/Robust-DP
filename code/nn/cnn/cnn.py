from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope, lazy_property
from nn.cnn.abccnn import ABCCNN

class CNN(ABCCNN):
    """
    Convolutional Neural Network
    """
    def __init__(self,
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[128, 128, 64, 64, 3], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.2, 0.2, 0.1],
                 # fc layers
                 fc_state_sizes=[1024], 
                 fc_leaky_ratio=[0.2, 0.2],
                 fc_drop_rate=[0, 0.75],
                 # output size
                 output_size=10,
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = None
                ):
        super().__init__(
            conv_filter_sizes, conv_strides, conv_padding, conv_channel_sizes, conv_leaky_ratio, img_channel, use_norm)
        # fc layers
        self.fc_state_sizes = fc_state_sizes 
        self.fc_leaky_ratio = fc_leaky_ratio
        self.fc_drop_rate = fc_drop_rate
        # output size
        self.output_size = output_size

        # shape of conv output
        self.conv_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.conv_channel_sizes[-1]]
        # attrs
        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.fc_weights, self.fc_biases, self.num_fc = self.fc_weights_biases()
        self.out_weights, self.out_biases = self.out_weights_biases()
        # property
        self.num_hidden_layers

    
    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv", "b_conv", self.conv_filter_sizes, self.img_channel, self.conv_channel_sizes)


    @lazy_method
    def fc_weights_biases(self):
        in_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        state_sizes = self.fc_state_sizes
        return self._fc_weights_biases("W_fc", "b_fc", in_size, state_sizes)


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
        
        for idx in range(num_layer):
            in_size = _func(in_size, state_sizes[idx], idx)

        return _weights, _biases, num_layer

    
    @lazy_method
    def out_weights_biases(self):
        in_size = self.fc_state_sizes[-1]
        state_size = [self.output_size]
        W, b, _ = self._fc_weights_biases("W_out", "b_out", in_size, state_size)
        return W, b

    
    @lazy_method
    def conv_layers(self, inputs, W_name="W_conv", b_name="b_conv"):
        net = tf.reshape(inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        layers_out = []
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
            layers_out.append(net)
            #net = ne.max_pool_2x2(net) # Pooling
        net = tf.identity(net, name='conv_output')
        #import pdb; pdb.set_trace()
        return net, layers_out


    @lazy_method
    def fc_layers(self, inputs, W_name="W_fc", b_name="b_fc"):
        net = tf.reshape(inputs, [-1, self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]])
        layers_out = []
        for layer_id in range(self.num_fc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.fc_weights[weight_name]
            curr_bias = self.fc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training, axis=1)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.enfc_leaky_ratio[layer_id], self.enfc_low_bound[layer_id], self.enfc_up_bound[layer_id]) # Nonlinear act
            net = ne.leaky_relu(net, self.fc_leaky_ratio[layer_id])
            net = ne.drop_out(net, self.fc_drop_rate[layer_id], self.is_training)
            layers_out.append(net)
            #net = ne.elu(net)

        net = tf.identity(net, name='fc_output')
        return net, layers_out

    
    @lazy_method
    def output_layer(self, inputs, W_name="W_out", b_name="b_out"):
        net = tf.reshape(inputs, [-1, self.fc_state_sizes[-1]])
        layer_id = 0
        weight_name = "{}{}".format(W_name, layer_id)
        bias_name = "{}{}".format(b_name, layer_id)
        curr_weight = self.out_weights[weight_name]
        curr_bias = self.out_biases[bias_name]
        logits = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
        preds = ne.sigmoid(logits)
        return preds, logits
    


    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        conv, convs_out = self.conv_layers(data)
        assert conv.get_shape().as_list()[1:] == self.conv_out_shape
        fc, fcs_out = self.fc_layers(conv)
        assert fc.get_shape().as_list()[1:] == [self.fc_state_sizes[-1]]
        preds, logits = self.output_layer(fc)
        self._layers_out = convs_out+fcs_out
        return preds, logits


    @lazy_method
    def hidden_layers_out(self):
        return self._layers_out
    

    @lazy_property
    def num_hidden_layers(self):
        return self.num_conv + self.num_fc
    

    def tf_load(self, sess, path, model_scope, name='cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, model_scope, name='cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.save(sess, path+'/'+name+spec)

        