""" Recurrent Encoder
"""
from dependency import *
from nn.rnn.abcrnn import ABCRNN
import utils.net_element as ne
from utils.tf_utils import *
from utils.decorator import lazy_method, lazy_method_no_scope


class RENC(ABCRNN):
    def __init__(self, 
                 # data info
                 max_time_steps=None,
                 # recurrent layers
                 rnn_state_sizes=[1024],
                 # encoder fc layers
                 enfc_state_sizes=[1024, 1024], 
                 enfc_leaky_ratio=[0.2, 0.2],
                 enfc_drop_rate=[0, 0.75],
                 # switch
                 is_reversed = False,
                 use_norm = None
                ):
        # data info
        self.max_time_steps = max_time_steps
        # recurrent layers
        self.rnn_state_sizes = rnn_state_sizes
        if len(rnn_state_sizes) == 1:
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_state_sizes[0])
            if is_reversed:
                self.rnn_cell_rvs = tf.nn.rnn_cell.LSTMCell(num_units=rnn_state_sizes[0])
        else:
            multi_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in rnn_state_sizes]
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(multi_cells)
            if is_reversed:
                self.rnn_cell_rvs = tf.nn.rnn_cell.MultiRNNCell(multi_cells)
        self.num_stacked = len(rnn_state_sizes)
        # encoder fc layers
        self.enfc_state_sizes = enfc_state_sizes 
        self.enfc_leaky_ratio = enfc_leaky_ratio
        self.enfc_drop_rate = enfc_drop_rate
        self.num_enfc = len(self.enfc_state_sizes[0])
        # switch
        self.is_reversed = is_reversed
        self.use_norm = use_norm

        self.enfc_h_weights, self.enfc_h_biases, self.enfc_c_weights, self.enfc_c_biases = self.enfc_complex_weights_biases()


    @lazy_method
    def enfc_weights_biases(self, sIdx=0):
        in_size = self.rnn_state_sizes[sIdx]
        if self.is_reversed:
            in_size *= 2
        state_sizes = self.enfc_state_sizes[sIdx]
        h_weights, h_biases, num_enfc = self._fc_weights_biases("W_enfc_s{}_h_".format(sIdx), "b_enfc_s{}_h_".format(sIdx), in_size, state_sizes)
        c_weights, c_biases, num_enfc = self._fc_weights_biases("W_enfc_s{}_c_".format(sIdx), "b_enfc_s{}_c_".format(sIdx), in_size, state_sizes)
        return h_weights, h_biases, c_weights, c_biases

    
    @lazy_method
    def enfc_complex_weights_biases(self):
        h_Ws = {}; h_bs = {}
        c_Ws = {}; c_bs = {}
        for sIdx in range(self.num_stacked):
            h_weights, h_biases, c_weights, c_biases = self.enfc_weights_biases(sIdx)
            h_Ws.update(h_weights); h_bs.update(h_biases)
            c_Ws.update(c_weights); c_bs.update(c_biases)
        return h_Ws, h_bs, c_Ws, c_bs


    @lazy_method
    def rnn_layer_(self, rnn_cell, inputs):
        if self.max_time_steps == None:
            max_time_steps, feature_size = inputs.get_shape().as_list()[1:]
            
        else:
            max_time_steps = self.max_time_steps
            feature_size = self.num_features
        
        inputs = tf.split(inputs, max_time_steps, axis = 1)
        state = rnn_cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
        
        h_state_list = []
        c_state_list = []
        for sIdx in range(self.num_stacked):
            h_state_list.append([])
            c_state_list.append([])
        for t in range(max_time_steps):
            data = tf.reshape(inputs[t], [-1, feature_size])
            out, state = rnn_cell(data, state)
            # first element is c state, second is h state -- 
            # line 1027 in https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/rnn_cell_impl.py
            if self.num_stacked > 1:
                for sIdx in range(self.num_stacked):
                    c_state_list[sIdx].append(state[sIdx][0])
                    h_state_list[sIdx].append(state[sIdx][1])
            else:
                c_state_list[0].append(state[0])
                h_state_list[0].append(state[1])

        c_states = []
        h_states = []
        for sIdx in range(self.num_stacked):
            c_states.append(tf.stack(c_state_list[sIdx], axis = 1))
            h_states.append(tf.stack(h_state_list[sIdx], axis = 1))
        return c_states, h_states

    
    @lazy_method
    def rnn_layer(self, inputs):
        c_states, h_states = self.rnn_layer_(self.rnn_cell, inputs)
        if self.is_reversed:
            inputs_rvs = tf.reverse(inputs, [1])
            c_states_, h_states_ = self.rnn_layer_(self.rnn_cell_rvs, inputs_rvs)
            c_states_all = []
            h_states_all = []
            if self.num_stacked > 1:
                for sIdx in range(self.num_stacked):
                    c_states_rvs = tf.reverse(c_states_[sIdx], [1])
                    h_states_rvs = tf.reverse(h_states_[sIdx], [1])
                    c_states_all.append(tf.concat([c_states[sIdx], c_states_rvs],2))
                    h_states_all.append(tf.concat([h_states[sIdx], h_states_rvs],2))
            else:
                c_states_all = tf.concat([c_states, c_states_rvs],2)
                h_states_all = tf.concat([h_states, h_states_rvs],2)
            c_states, h_states = c_states_all, h_states_all

        return c_states, h_states


    @lazy_method
    def complex_fc_layers(self, inputs, weights_dict, biases_dict, fc_leaky_ratio, fc_drop_rate, num_stacked, num_fc, W_name, b_name, fc_type):
        nets = inputs; outs = []
        for s_id in range(num_stacked):
            net = nets[s_id]
            sizes = net.get_shape().as_list()[1:]
            net = tf.reshape(net, [-1, sizes[-1]])
            weight_name = "{}s{}_{}_".format(W_name, s_id, fc_type)
            bias_name = "{}s{}_{}_".format(b_name, s_id, fc_type)
            net = self._fc_layers(net, weights_dict, biases_dict, fc_leaky_ratio, fc_drop_rate, num_fc, weight_name, bias_name)
            net = tf.reshape(net, [-1]+sizes[:-1]+[net.get_shape().as_list()[-1]])
            outs.append(net)
        return outs

    
    @lazy_method
    def enfc_layers(self, c_inputs, h_inputs, W_name="W_enfc_", b_name="b_enfc_"):
        #import pdb; pdb.set_trace()
        c_outs = self.complex_fc_layers(c_inputs, self.enfc_c_weights, self.enfc_c_biases, 
                                        self.enfc_leaky_ratio, self.enfc_drop_rate,
                                        self.num_stacked, self.num_enfc, W_name, b_name, "c")
        h_outs = self.complex_fc_layers(h_inputs, self.enfc_h_weights, self.enfc_h_biases, 
                                        self.enfc_leaky_ratio, self.enfc_drop_rate,
                                        self.num_stacked, self.num_enfc, W_name, b_name, "h")
        return c_outs, h_outs


    @lazy_method
    def evaluate(self, inputs, is_training):
        self.is_training = is_training
        self.num_features = inputs.get_shape().as_list()[-1]
        c_states, h_states = self.rnn_layer(inputs)
        c_enfcs, h_enfcs = self.enfc_layers(c_states, h_states)
        return c_enfcs, h_enfcs, h_states


    def tf_load(self, sess, path, model_scope, name='deep_renc.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, model_scope, name='deep_renc.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.save(sess, path+'/'+name+spec)
