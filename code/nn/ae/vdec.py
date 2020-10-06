""" Variational Convolutional Decoder
"""
from nn.ae.cdec import CDEC
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_property, lazy_method_no_scope
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


class VDEC(CDEC):
    def __init__(self, 
                 vtype,
                 output_low_bound,
                 output_up_bound,
                 # deconv layers
                 decv_filter_sizes=[3,3], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 decv_strides = [1,1], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 decv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 decv_channel_sizes=[3, 32, 64, 128, 256],  # [1, 128, 128, 128, 128]
                 decv_leaky_ratio=[0.1, 0.2, 0.2, 0.4, 0.4],
                 # bottleneck
                 central_state_size=2048, 
                 # decoder fc layers
                 defc_state_sizes=[512, 1024],
                 defc_leaky_ratio=[0.2, 0.2, 0.2],
                 defc_drop_rate=[0, 0, 0],
                 # img channel
                 img_channel=None,
                 # switch
                 use_norm = False
                ):
        self.vtype = vtype
        super().__init__(output_low_bound, output_up_bound, decv_filter_sizes, decv_strides, decv_padding, decv_channel_sizes, decv_leaky_ratio,
              central_state_size, defc_state_sizes, defc_leaky_ratio, defc_drop_rate, 
              img_channel, use_norm
             )
    
    
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
    def kl_distance(self):
        if self.vtype == "gauss":
            self.prior = tf.distributions.Normal(tf.zeros(self.central_state_size), tf.ones(self.central_state_size))
            self.kl = self.central_distribution.kl_divergence(self.prior)
            loss_kl = tf.reduce_mean(tf.reduce_sum(self.kl, axis=1))
        elif self.vtype == 'vmf':
            self.prior = HypersphericalUniform(self.central_state_size-1, dtype=tf.float32)
            self.kl = self.central_distribution.kl_divergence(self.prior)
            loss_kl = tf.reduce_mean(self.kl)
        else:
            raise NotImplemented
        return loss_kl

    
    def tf_load(self, sess, path, scope, name='deep_vdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_vdec.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
