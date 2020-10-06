""" Recurrent Neural Network
"""
from nn.rnn.abcrnn import ABCRNN

from dependency import *
import utils.net_element as ne
from utils.tf_utils import *
from utils.decorator import lazy_method, lazy_method_no_scope


class RNN(ABCRNN):
    def __init__(self, 
                 # data info
                 max_time_steps=None,
                 # recurrent layers
                 rnn_state_sizes=[64],
                 # encoder fc layers
                 fc_state_sizes=[[64]], 
                 fc_leaky_ratio=[0.2],
                 fc_drop_rate=[0.25],
                 # out fc layer
                 out_state_size=1,
                 out_leaky_ratio=0.2,
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
        self.fc_state_sizes = fc_state_sizes 
        self.fc_leaky_ratio = fc_leaky_ratio
        self.fc_drop_rate = fc_drop_rate
        self.num_fc = len(self.fc_state_sizes[0])
        # out fc layer
        self.out_state_size = out_state_size
        self.out_leaky_ratio = out_leaky_ratio
        # switch
        self.is_reversed = is_reversed
        self.use_norm = use_norm

        self.fc_h_weights, self.fc_h_biases = self.fc_complex_weights_biases()
        self.out_h_weight, self.out_h_bias = self.out_complex_weights_biases()


    @lazy_method
    def fc_weights_biases(self, sIdx=0):
        in_size = self.rnn_state_sizes[sIdx]
        if self.is_reversed:
            in_size *= 2
        state_sizes = self.fc_state_sizes[sIdx]
        h_weights, h_biases, num_fc = self._fc_weights_biases("W_fc_s{}_h_".format(sIdx), "b_fc_s{}_h_".format(sIdx), in_size, state_sizes)
        #c_weights, c_biases, num_fc = self._fc_weights_biases("W_fc_s{}_c_".format(sIdx), "b_fc_s{}_c_".format(sIdx), in_size, state_sizes)
        return h_weights, h_biases

    
    @lazy_method
    def fc_complex_weights_biases(self):
        h_Ws = {}; h_bs = {}
        for sIdx in range(self.num_stacked):
            h_weights, h_biases = self.fc_weights_biases(sIdx)
            h_Ws.update(h_weights); h_bs.update(h_biases)
        return h_Ws, h_bs


    @lazy_method
    def out_weight_bias(self, sIdx):
        in_size = self.rnn_state_sizes[sIdx]
        state_sizes = [self.out_state_size]
        h_weights, h_biases, num_defc = self._fc_weights_biases("W_out_s{}_h_".format(sIdx), "b_out_s{}_h_".format(sIdx), in_size, state_sizes)
        #c_weights, c_biases, num_defc = self._fc_weights_biases("W_out_s{}_c_".format(sIdx), "b_out_s{}_c_".format(sIdx), in_size, state_sizes)
        return h_weights, h_biases
    

    @lazy_method
    def out_complex_weights_biases(self):
        h_Ws = {}; h_bs = {}
        for sIdx in range(self.num_stacked):
            h_weights, h_biases = self.out_weight_bias(sIdx)
            h_Ws.update(h_weights); h_bs.update(h_biases)
        return h_Ws, h_bs


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
    def fc_layers(self, h_inputs, W_name="W_fc_", b_name="b_fc_"):
        #import pdb; pdb.set_trace()
        #c_outs = self.complex_fc_layers(c_inputs, self.fc_c_weights, self.fc_c_biases, self.num_stacked, self.num_fc, W_name, b_name, "c")
        h_outs = self.complex_fc_layers(h_inputs, self.fc_h_weights, self.fc_h_biases, 
                                        self.fc_leaky_ratio, self.fc_drop_rate,
                                        self.num_stacked, self.num_fc, W_name, b_name, "h")
        return h_outs


    @lazy_method
    def out_layers(self, h_inputs, W_name="W_out_", b_name="b_out_"):
        h_nets = h_inputs
        h_outs = self.complex_fc_layers(h_inputs, self.out_h_weight, self.out_h_bias, 
                                        [self.out_leaky_ratio], [0.0],
                                        self.num_stacked, 1, W_name, b_name, "h")
        logits = h_outs[-1]
        preds = ne.sigmoid(logits)
        return preds, logits
    

    @lazy_method
    def evaluate(self, inputs, is_training):
        self.is_training = is_training
        self.num_features = inputs.get_shape().as_list()[-1]
        _, h_states = self.rnn_layer(inputs)
        h_state = get_latest_tensor(h_states, self.max_time_steps)
        h_fc = self.fc_layers(h_state)
        preds, logits = self.out_layers(h_fc)
        return preds, logits


    def tf_load(self, sess, path, model_scope, name='deep_rnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, model_scope, name='deep_rnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.save(sess, path+'/'+name+spec)
