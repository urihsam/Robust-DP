""" Recurrent Decoder
"""
from dependency import *
from nn.rnn.abcrnn import ABCRNN
import utils.net_element as ne
from utils.tf_utils import *
from utils.decorator import lazy_method, lazy_method_no_scope


class RDEC(ABCRNN):
    def __init__(self, 
                 # data info
                 max_time_steps=None,
                 # recurrent layers
                 attention_size = None,
                 memory = None,
                 rnn_state_sizes=[1024],
                 # decoder fc layers
                 defc_state_sizes=[1024], 
                 defc_leaky_ratio=[0.2, 0.2],
                 defc_drop_rate=[0, 0.75],
                 # out fc layer
                 out_state_size=1024,
                 out_leaky_ratio=0.2,
                 # switch
                 use_norm = None
                ):
        # data info
        self.max_time_steps = max_time_steps
        # recurrent layers
        if attention_size == None:
            self.use_attention = False
        else:
            self.use_attention = True
        self.attention_size = attention_size
        self.memory = memory
        self.rnn_state_sizes = rnn_state_sizes
        if len(rnn_state_sizes) == 1:
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_state_sizes[0])
        else:
            multi_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in rnn_state_sizes]
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(multi_cells)
        self.num_stacked = len(rnn_state_sizes)
        # encoder fc layers
        self.defc_state_sizes = defc_state_sizes 
        self.defc_leaky_ratio = defc_leaky_ratio
        self.defc_drop_rate = defc_drop_rate
        self.num_defc = len(self.defc_state_sizes[0])
        # bout fc layer
        self.out_state_size = out_state_size
        self.out_leaky_ratio = out_leaky_ratio
        # switch
        self.use_norm = use_norm

        self.defc_h_weights, self.defc_h_biases, self.defc_c_weights, self.defc_c_biases = self.defc_complex_weights_biases()
        self.out_h_weight, self.out_h_bias = self.out_complex_weights_biases()


    @lazy_method
    def defc_weights_biases(self, sIdx=0):
        in_size = self.defc_state_sizes[sIdx][0]
        state_sizes = self.defc_state_sizes[sIdx][1:] + [self.rnn_state_sizes[sIdx]]
        h_weights, h_biases, num_defc = self._fc_weights_biases("W_defc_s{}_h_".format(sIdx), "b_defc_s{}_h_".format(sIdx), in_size, state_sizes)
        c_weights, c_biases, num_defc = self._fc_weights_biases("W_defc_s{}_c_".format(sIdx), "b_defc_s{}_c_".format(sIdx), in_size, state_sizes)
        return h_weights, h_biases, c_weights, c_biases

    
    @lazy_method
    def defc_complex_weights_biases(self):
        h_Ws = {}; h_bs = {}
        c_Ws = {}; c_bs = {}
        for sIdx in range(self.num_stacked):
            h_weights, h_biases, c_weights, c_biases = self.defc_weights_biases(sIdx)
            h_Ws.update(h_weights); h_bs.update(h_biases)
            c_Ws.update(c_weights); c_bs.update(c_biases)
        return h_Ws, h_bs, c_Ws, c_bs

    
    @lazy_method
    def out_weight_bias(self):
        in_size = self.rnn_state_sizes[self.num_stacked-1]
        if self.use_attention:
            in_size *= 2
        state_sizes = [self.out_state_size]
        h_weights, h_biases, num_defc = self._fc_weights_biases("W_out_s{}_h_".format(0), "b_out_s{}_h_".format(0), in_size, state_sizes)
        return h_weights, h_biases
    

    @lazy_method
    def out_complex_weights_biases(self):
        h_Ws = {}; h_bs = {}
        h_weights, h_biases= self.out_weight_bias()
        h_Ws.update(h_weights); h_bs.update(h_biases)
        return h_Ws, h_bs


    @lazy_method
    def rnn_layer(self, rnn_cell, inputs=None, init_state=None):
        if self.max_time_steps == None:
            max_time_steps, feature_size = inputs.get_shape().as_list()[1:]
        else:
            max_time_steps = self.max_time_steps
            feature_size = self.num_features
        if inputs == None:
            inputs = tf.constant(0.0, shape=(FLAGS.BATCH_SIZE, max_time_steps, feature_size))
        inputs = tf.split(inputs, max_time_steps, axis = 1)
        if init_state == None:
            state = rnn_cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
        else:
            state = init_state
        
        h_state_list = []
        c_state_list = []
        out_list = []
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
                    c_state_list[sIdx].append(state[0][sIdx][0])
                    h_state_list[sIdx].append(state[0][sIdx][1])
            else:
                c_state_list[0].append(state[0][0])
                h_state_list[0].append(state[0][1])
            out_list.append(out)
        c_states = []
        h_states = []
        for sIdx in range(self.num_stacked):
            c_states.append(tf.stack(c_state_list[sIdx], axis = 1))
            h_states.append(tf.stack(h_state_list[sIdx], axis = 1))
        out_states = tf.stack(out_list, axis = 1)  
        return c_states, h_states, out_states

    
    @lazy_method
    def att_layer(self, rnn_cell, memory):
        mechanism = tf.contrib.seq2seq.BahdanauAttention(self.attention_size, memory)
        att_cell = tf.contrib.seq2seq.AttentionWrapper(rnn_cell, mechanism)
        return att_cell


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
    def defc_layers(self, c_inputs, h_inputs, W_name="W_defc_", b_name="b_defc_"):
        c_nets = c_inputs; h_nets = h_inputs
        c_outs = self.complex_fc_layers(c_inputs, self.defc_c_weights, self.defc_c_biases,
                                        self.defc_leaky_ratio, self.defc_drop_rate,
                                        self.num_stacked, self.num_defc, W_name, b_name, "c")
        h_outs = self.complex_fc_layers(h_inputs, self.defc_h_weights, self.defc_h_biases, 
                                        self.defc_leaky_ratio, self.defc_drop_rate,
                                        self.num_stacked, self.num_defc, W_name, b_name, "h")
        return c_outs, h_outs
    

    @lazy_method
    def out_layers(self, h_inputs, W_name="W_out_", b_name="b_out_"):
        h_nets = h_inputs
        h_outs = self.complex_fc_layers(h_inputs, self.out_h_weight, self.out_h_bias, 
                                        [self.out_leaky_ratio], [0.0],
                                        1, 1, W_name, b_name, "h")
        return h_outs[-1]


    @lazy_method
    def evaluate(self, c_inputs, h_inputs, is_training):
        self.is_training = is_training
        self.num_features = FLAGS.DEC_INPUTS_SIZE
        c_state = get_latest_tensor(c_inputs, self.max_time_steps)
        h_state = get_latest_tensor(h_inputs, self.max_time_steps)
        c_defc, h_defc = self.defc_layers(c_state, h_state)
        init_state = convert_to_tuple(c_defc, h_defc)
        if self.use_attention:
            self.att_cell = self.att_layer(self.rnn_cell, self.memory)
            state = self.att_cell.zero_state(FLAGS.BATCH_SIZE, tf.float32)
            state = state.clone(cell_state=(init_state[0], init_state[1]))
            c_states, h_states, out = self.rnn_layer(self.att_cell, None, state)
        else:
            c_states, h_states, out = self.rnn_layer(self.rnn_cell, None, init_state)
        h_outs = self.out_layers([out])
        return h_outs


    def tf_load(self, sess, path, model_scope, name='deep_rdec.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, model_scope, name='deep_rdec.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope))
        saver.save(sess, path+'/'+name+spec)
