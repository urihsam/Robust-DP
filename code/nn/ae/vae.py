""" Variational Convolutional Autoconvder
"""
from nn.ae.cae import CAE
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_property, lazy_method_no_scope
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


class VAE(CAE):
    def __init__(self, 
                 vtype,
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
        self.vtype = vtype
        super().__init__(output_low_bound, output_up_bound, nonlinear_low_bound, nonlinear_up_bound,
              conv_filter_sizes, conv_strides, conv_padding, conv_channel_sizes, conv_leaky_ratio,
              decv_filter_sizes, decv_strides, decv_padding, decv_channel_sizes, decv_leaky_ratio,
              enfc_state_sizes, enfc_leaky_ratio, enfc_drop_rate, central_state_size, 
              defc_state_sizes, defc_leaky_ratio, defc_drop_rate, 
              img_channel, use_norm
             )

    
    @lazy_method
    def enfc_weights_biases(self):
        in_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        state_sizes = self.enfc_state_sizes + [self.central_state_size]
        return self._fc_weights_biases("W_enfc", "b_enfc", in_size, state_sizes, sampling=True)

    
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
                raise NotImplemented
        else:
             _func(in_size, state_sizes[num_layer-1], num_layer-1)
        #import pdb; pdb.set_trace()

        return _weights, _biases, num_layer


    @lazy_method
    def enfc_layers(self, inputs, W_name="W_enfc", b_name="b_enfc"):
        net = tf.reshape(inputs, [-1, self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]])
        def _func(net, layer_id, postfix="", act_func="leaky"):
            weight_name = "{}{}{}".format(W_name, layer_id, postfix)
            bias_name = "{}{}{}".format(b_name, layer_id, postfix)
            curr_weight = self.enfc_weights[weight_name]
            curr_bias = self.enfc_biases[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training, axis=1)
            elif self.use_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.enfc_leaky_ratio[layer_id], self.enfc_low_bound[layer_id], self.enfc_up_bound[layer_id]) # Nonlinear act
            if act_func=="leaky":
                net = ne.leaky_relu(net, self.enfc_leaky_ratio[layer_id])
            elif act_func=="soft":
                net = tf.nn.softplus(net)
            #net = ne.drop_out(net, self.enfc_drop_rate[layer_id], self.is_training)
            return net

        for layer_id in range(self.num_enfc-1):
            net = _func(net, layer_id)
        # Last layer
        if self.vtype == "gauss":
            # compute mean and log of var of the normal distribution
            """net_mu = tf.minimum(tf.maximum(-5.0, _func(net, self.num_enfc-1, "_mu")), 5.0)
            ## Set low and up bounds for log_sigma_sq
            '''net_log_sigma_sq = tf.minimum(tf.maximum(-10.0, _func(net, self.num_enfc-1, "_sigma")), 5.0)
            net_sigma = tf.sqrt(tf.exp(net_log_sigma_sq))'''
            net_sigma = tf.maximum(_func(net, self.num_enfc-1, "_sigma", "soft"), 5.0)"""
            net_mu = _func(net, self.num_enfc-1, "_mu")
            net_log_sigma_sq = tf.minimum(tf.maximum(-10.0, _func(net, self.num_enfc-1, "_sigma")), 5.0)
            net_sigma = tf.sqrt(tf.exp(net_log_sigma_sq))
        elif self.vtype == "vmf":
            # compute mean and log of var of the von Mises-Fisher
            #net_mu = tf.minimum(tf.maximum(0.0, _func(net, self.num_enfc-1, "_mu", None)), 0.0)
            net_mu = _func(net, self.num_enfc-1, "_mu", None)
            net_mu = tf.nn.l2_normalize(net_mu, axis=-1)
            #net_mu = tf.nn.l2_normalize(_func(net, self.num_enfc-1, "_mu"), axis=1)
            ## Set low and up bounds for log_sigma_sq
            #net_log_sigma_sq = tf.minimum(tf.maximum(0.0, _func(net, self.num_enfc-1, "_log_sigma_sq")), 10.0)
            net_sigma = _func(net, self.num_enfc-1, "_sigma", "soft") + 200.0
        else:
            raise NotImplemented

        net_mu = tf.identity(net_mu, name="output_mu")
        net_sigma = tf.identity(net_sigma, name="output_sigma")
        return net_mu, net_sigma
    
    
    @lazy_method
    def encoder(self, inputs):
        conv = self.conv_layers(inputs)
        assert conv.get_shape().as_list()[1:] == self.conv_out_shape
        self.central_mu, self.central_sigma = self.enfc_layers(conv)
        if self.vtype == "gauss":
            assert self.central_mu.get_shape().as_list()[1:] == [self.central_state_size]
        elif self.vtype == "vmf":
            assert self.central_sigma.get_shape().as_list()[1:] == [1]
        
        """# epsilon
        eps = tf.random_normal(tf.shape(self.central_mu), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        enfc = tf.add(self.central_mu, tf.multiply(tf.sqrt(tf.exp(self.central_log_sigma_sq)), eps))"""
        if self.vtype == "gauss":
            self.central_distribution = tf.distributions.Normal(self.central_mu, self.central_sigma)
        elif self.vtype == "vmf":
            self.central_distribution = VonMisesFisher(self.central_mu, self.central_sigma)
        self.central_states = self.central_distribution.sample()
        return self.central_states

    
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
    
    @lazy_method
    def gauss_kl_distance(self):
        loss = -0.5 * tf.reduce_sum(
            1 + self.central_log_sigma_sq - tf.square(self.central_mu) - tf.exp(self.central_log_sigma_sq), 1)
            
        return loss

    
    def tf_load(self, sess, path, name='deep_vcae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, name='deep_vcae.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder'))
        saver.save(sess, path+'/'+name+spec)
