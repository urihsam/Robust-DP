from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope, lazy_property
from nn.cnn.abccnn import ABCCNN

class RESCNN(ABCCNN):
    """
    Convolutional Neural Network
    """
    def __init__(self,
                 # conv layers
                 conv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[128, 256, 512], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4],
                 # residual layers
                 num_residual=4,
                 res_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 res_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_channel_sizes=[512, 512],
                 res_leaky_ratio=[0.2, 0.2],
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = None
                ):
        super().__init__(
            conv_filter_sizes, conv_strides, conv_padding, conv_channel_sizes, conv_leaky_ratio, img_channel, use_norm)
        # residual layers
        self.num_residual = num_residual
        self.res_filter_sizes = res_filter_sizes
        self.res_strides = res_strides
        self.res_padding = res_padding
        self.res_channel_sizes = res_channel_sizes
        self.res_leaky_ratio = res_leaky_ratio

        # shape of conv output
        self.conv_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.conv_channel_sizes[-1]]
        self.res_out_shape = [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.res_channel_sizes[-1]]
        # attrs
        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        self.res_filters, self.res_biases, self.num_res = self.conv_weights_biases()
        # property
        self.num_hidden_layers

    
    @lazy_method
    def conv_weights_biases(self):
        return self._conv_weights_biases("W_conv_", "b_conv_", self.conv_filter_sizes, self.img_channel, self.conv_channel_sizes)


    @lazy_method
    def res_weights_biases(self):
        Ws = {}; bs = {}; num_res=[]
        in_channel = self.conv_channel_sizes[-1]
        for idx in self.num_residual:
            W_name = "W_res{}_".format(idx)
            b_name = "b_res{}_".format(idx)
            W_, b_, n_res = self._conv_weights_biases(W_name, b_name, self.res_filter_sizes, in_channel, self.res_channel_sizes)
            Ws.update(W_)
            bs.update(b_)
            num_res.append(n_res)
            in_channel = self.res_channel_sizes[-1]
        return Ws, bs

    
    @lazy_method
    def conv_layers(self, inputs, W_name="W_conv_", b_name="b_conv_"):
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
    def res_layers(self, inputs, W_name="W_res", b_name="b_res"):
        net = tf.reshape(inputs, [-1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, self.img_channel])
        layers_out = []
        for res_id in range(self.num_residual):
            res_net = net
            for layer_id in range(self.num_res[res_id]):
                filter_name = "{}{}_{}".format(W_name, res_id, layer_id)
                bias_name = "{}{}_{}".format(b_name, res_id, layer_id)
                curr_filter = self.res_filters[filter_name]
                curr_bias = self.res_biases[bias_name]
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
            net += res_net
            layers_out.append(net)
        net = tf.identity(net, name='conv_output')
        #import pdb; pdb.set_trace()
        return net, layers_out


    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        conv, convs_out = self.conv_layers(data)
        assert conv.get_shape().as_list()[1:] == self.conv_out_shape
        res, res_out = self.res_layers(conv)
        assert res.get_shape().as_list()[1:] == self.res_out_shape
        return res


    @lazy_method
    def hidden_layers_out(self): # one out for each block of residual
        return self._layers_out
    

    @lazy_property
    def num_hidden_layers(self): # one block of residual as one layer
        return self.num_conv + self.num_residual
    

    def tf_load(self, sess, path, scope, name='rescnn.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='rescnn.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)

        