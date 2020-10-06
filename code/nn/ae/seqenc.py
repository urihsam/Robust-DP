"""
encoder for sequence data, e.g., time series data.
"""
from dependency import *
import utils.net_element as ne
from utils.decorator import lazy_method, lazy_method_no_scope


class SEQENC:
    def __init__(self, 
                 feature_size, 
                 memory=None, 
                 num_blocks=2,
                 num_att_header=8,
                 leaky_ratio=0.2,
                 drop_rate=0.2,
                 use_norm = None):
        self.feature_size = feature_size
        self.memory = memory
        if self.memory != None:
            self.memory_size = self.memory.get_shape().as_list()[-1]
        else:
            self.memory_size = feature_size
        self.num_blocks = num_blocks
        self.num_att_header = num_att_header
        self.leaky_ratio = leaky_ratio
        self.drop_rate = drop_rate
        self.use_norm = use_norm
        ## weights
        self.att_weights, self.att_biases, self.att_gammas = self.att_weights_biases()


    @lazy_method
    def att_weights_biases(self):
        Ws = {}; bs = {}; gammas = {}
        for idx in range(self.num_blocks):
            W_Q, b_Q, _ = self._fc_weights_biases("W_att_Q_l{}_".format(idx), 
                            "b_att_Q_l{}_".format(idx), self.feature_size, self.feature_size)
            W_K, b_K, _ = self._fc_weights_biases("W_att_K_l{}_".format(idx), 
                            "b_att_K_l{}_".format(idx), self.memory_size, self.feature_size)
            W_V, b_V, _ = self._fc_weights_biases("W_att_V_l{}_".format(idx),
                            "b_att_V_l{}_".format(idx), self.memory_size, self.feature_size)
            gamma["g_att_l{}".format(idx)] = tf.get_variable("Gamma_att_l{}".format(idx), 
                            [1], initializer=tf.constant_initializer(0.0))
            Ws.update(W_Q); Ws.update(W_K); Ws.update(W_V); 
            bs.update(b_Q); bs.update(b_K); bs.update(b_V); 
            gammas.update(gamma)
        return Ws, bs, gammas


    @lazy_method
    def _fc_weights_biases(self, W_name, b_name, in_size, state_sizes, init_type="HE", no_bias=False):
        num_layer = len(state_sizes)
        _weights = {}
        _biases = {}
        def _func(in_size, out_size, idx, postfix=""):
            W_key = "{}{}{}".format(W_name, idx, postfix)
            W_shape = [in_size, out_size]
            _weights[W_key] = ne.weight_variable(W_shape, name=W_key, init_type=init_type)

            if no_bias == False:
                b_key = "{}{}{}".format(b_name, idx, postfix)
                b_shape = [out_size]
                _biases[b_key] = ne.bias_variable(b_shape, name=b_key)

            in_size = out_size

            # tensorboard
            tf.summary.histogram("Weight_"+W_key, _weights[W_key])
            if no_bias == False:
                tf.summary.histogram("Bias_"+b_key, _biases[b_key])
            
            return in_size
        
        for idx in range(num_layer):
            in_size = _func(in_size, state_sizes[idx], idx)

        return _weights, _biases, num_layer


    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self._pad_id), tf.float32)  # mask '<pad>'
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
        return mask


    @lazy_method
    def _multihead_attention_layer(self, layer_idx, query, memory=None, mask=None):
        if memory is None:
            memory = query

        # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
        Q = ne.fully_conn(query, self.att_weights["W_att_Q_l{}_0".format(layer_idx)], 
                    self.att_biases["b_att_Q_l{}_0".format(layer_idx)])
        Q = ne.leaky_relu(Q, self.leaky_ratio[layer_idx])

        K = ne.fully_conn(memory, self.att_weights["W_att_K_l{}_0".format(layer_idx)], 
                    self.att_biases["b_att_K_l{}_0".format(layer_idx)])
        K = ne.leaky_relu(K, self.leaky_ratio[layer_idx])

        V = ne.fully_conn(memory, self.att_weights["W_att_V_l{}_0".format(layer_idx)], 
                    self.att_biases["b_att_V_l{}_0".format(layer_idx)])
        V = ne.leaky_relu(V, self.leaky_ratio[layer_idx])

        # Split the matrix to multiple heads and then concatenate to have a larger
        # batch size: [h*batch, q_size/k_size, d_model/num_heads]
        Q_split = tf.concat(tf.split(Q, self.num_att_header, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, self.num_att_header, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, self.num_att_header, axis=2), axis=0)
        if mask != None:
            mask = tf.tile(mask, [self.num_att_header, 1, 1])


        # Apply scaled dot product attention
        d = self.feature_size // self.num_att_header
        assert d == Q_split.shape[-1] == K_split.shape[-1] == V_split.shape[-1]

        out = tf.matmul(Q_split, tf.transpose(K_split, [0, 2, 1]))  # [h*batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = ne.softmax(out)  # [h * batch, q_size, k_size]
        out = ne.dropout(out, self.drop_rate[layer_idx], self.is_training)
        out = tf.matmul(out, V_split)  # [h * batch, q_size, d_model]

        # Merge the multi-head back to the original shape
        out = tf.concat(tf.split(out, self.num_att_header, axis=0), axis=2)  # [bs, q_size, d_model]

        return out

    
    @lazy_method
    def multihead_att_layer(self, query, memory=None, mask=None):
        net = query
        for idx in range(self.num_blocks):
            net = self._multihead_attention_layer(idx, net, memory, mask)
        return net

    
    @lazy_method
    def evaluate(self, data, is_training):
        self.is_training = is_training
        input_mask = self.construct_padding_mask(data)
        seq_att = self.multihead_att_layer(data, mask=input_mask)
        return seq_att


    def tf_load(self, sess, path, scope, name='deep_seqenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='deep_seqenc.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
