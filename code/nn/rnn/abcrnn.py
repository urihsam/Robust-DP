""" Recurrent Network
"""
import abc
from abc import ABC
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_method_no_scope


class ABCRNN(ABC):
    def __init__(self, 
                 # data info
                 max_time_steps=None,
                 # recurrent layers
                 rnn_state_sizes=[1024],
                 # encoder fc layers
                 enfc_state_sizes=[1024], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # switch
                 use_norm = None
                ):
        # recurrent layers
        self.rnn_state_sizes = rnn_state_sizes
        if len(rnn_state_sizes) == 1:
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_state_sizes[0])
        else:
            multi_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in rnn_state_sizes]
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(multi_cells)
        self.num_stacked = len(rnn_state_sizes)
        # encoder fc layers
        self.enfc_state_sizes = enfc_state_sizes 
        self.enfc_leaky_ratio = enfc_leaky_ratio
        self.enfc_drop_rate = enfc_drop_rate
        self.num_enfc = len(self.enfc_state_sizes)
        # data info
        self.max_time_steps = max_time_steps
        # switch
        self.use_norm = use_norm

        self.enfc_h_weights, self.enfc_h_biases, self.enfc_c_weights, self.enfc_c_biases = self.enfc_complex_weights_biases()


    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        def _func(in_size, out_size, idx):
            W_key = "{}{}".format(W_name, idx)
            W_shape = [in_size, out_size]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key)

            b_key = "{}{}".format(b_name, idx)
            b_shape = [out_size]
            _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = out_size

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            tf.summary.histogram("Bias_"+b_key, _biases[b_key])
            
            return in_size
        
        for idx in range(num_layer):
            in_size = _func(in_size, state_sizes[idx], idx)
        #import pdb; pdb.set_trace()

        return _weights, _biases, num_layer


    @abc.abstractmethod
    def rnn_layer(self, rnn_cell, inputs, init_state=None):
        pass


    @lazy_method
    def _fc_layers(self, inputs, weights_dict, biases_dict, fc_leaky_ratio, fc_drop_rate, num_fc, W_name, b_name):
        net = inputs
        for layer_id in range(num_fc):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = weights_dict[weight_name]
            curr_bias = biases_dict[bias_name]
            net = ne.fully_conn(net, weights=curr_weight, biases=curr_bias)
            # batch normalization
            if self.use_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training, axis=-1)
            #net = ne.leaky_brelu(net, self.enfc_leaky_ratio[layer_id], self.enfc_low_bound[layer_id], self.enfc_up_bound[layer_id]) # Nonlinear act
            net = ne.leaky_relu(net, fc_leaky_ratio[layer_id])
            net = ne.drop_out(net, fc_drop_rate[layer_id], self.is_training)
            #net = ne.elu(net)

        net = tf.identity(net, name='output')
        return net


    @abc.abstractmethod
    def evaluate(self, inputs, init_state, memory, is_training):
        pass

