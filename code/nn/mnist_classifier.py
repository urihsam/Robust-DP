from dependency import *
from utils.decorator import lazy_method, lazy_method_no_scope, lazy_property
from nn.ae.resenc_v2 import RESENC


class MNISTCNN(RESENC):
    def __init__(self, 
                 # conv layers
                 conv_filter_sizes=[4,4], #[[3,3], [3,3], [3,3], [3,3], [3,3]], 
                 conv_strides = [2,2], #[[1,1], [1,1], [1,1], [1,1], [1,1]],
                 conv_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 conv_channel_sizes=[32, 64, 128], # [128, 128, 128, 128, 1]
                 conv_leaky_ratio=[0.4, 0.4, 0.4],
                 conv_drop_rate=[0.4, 0.4, 0.4],
                 conv_residual=False,
                 # residual layers
                 num_res_block=4,
                 res_block_size=2,
                 res_filter_sizes=[1,1], 
                 res_strides = [1,1],
                 res_padding = "SAME", #["SAME", "SAME", "SAME", "SAME", "SAME"],
                 res_leaky_ratio=0.2,
                 res_drop_rate=0.4,
                 # out fc layer
                 out_state=196,
                 out_fc_states=[10],
                 out_leaky_ratio=0.2,
                 # img channel
                 img_channel=None,
                 # switch
                 out_norm = None,
                 use_norm = None,
                 include_top=True
                ):
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
        self.conv_drop_rate = conv_drop_rate
        self.conv_residual = conv_residual
        # res layers
        self.num_res_block = num_res_block
        if self.num_res_block != 0:
            self.res_block_size = res_block_size
            if isinstance(res_filter_sizes[0], list):
                self.res_filter_sizes = res_filter_sizes
            else:
                self.res_filter_sizes = [res_filter_sizes] * self.res_block_size
            if isinstance(res_strides[0], list):
                self.res_strides = res_strides
            else:
                self.res_strides = [res_strides] * self.res_block_size
            if isinstance(res_padding, str):
                self.res_padding = [res_padding] * self.res_block_size
            else:
                self.res_padding = res_padding
            if isinstance(res_leaky_ratio, list):
                self.res_leaky_ratio = res_leaky_ratio
            else:
                self.res_leaky_ratio = [res_leaky_ratio] * self.res_block_size
            if isinstance(res_drop_rate, list):
                self.res_drop_rate = res_drop_rate
            else:
                self.res_drop_rate = [res_drop_rate] * self.res_block_size
        self.include_top = include_top
        if self.include_top:
            # out fc layer
            self.out_state = out_state
            if isinstance(out_fc_states, list):
                self.out_fc_states = out_fc_states
            else:
                self.out_fc_states = [out_fc_states]
            self.out_leaky_ratio = out_leaky_ratio
        # img channel
        if img_channel == None:
            self.img_channel = FLAGS.NUM_CHANNELS
        else:
            self.img_channel = img_channel
        # switch
        self.out_norm = out_norm
        self.use_norm = use_norm
        self.use_class_label = False


        self.conv_filters, self.conv_biases, self.num_conv = self.conv_weights_biases()
        if self.num_res_block != 0:
            self.res_filters, self.res_biases = self.res_weights_biases()
        if self.include_top:
            self.out_weight, self.out_bias = self.out_weight_bias()
    

    def out_layer(self, inputs, W_name="W_out_", b_name="b_out_"):
        net = inputs
        h, w, c = net.get_shape().as_list()[1:]
        print(h, w, c)
        assert h*w*c == self.out_state
        net = tf.reshape(net, [-1, self.out_state])
        
        for layer_id in range(len(self.out_fc_states)):
            weight_name = "{}{}".format(W_name, layer_id)
            bias_name = "{}{}".format(b_name, layer_id)
            curr_weight = self.out_weight[weight_name]
            curr_bias = self.out_bias[bias_name]

            net = ne.fully_conn(net, curr_weight, curr_bias)
            if self.out_norm == "BATCH":
                net = ne.batch_norm(net, self.is_training)
            elif self.out_norm == "LAYER":
                net = ne.layer_norm(net, self.is_training)
            #net = ne.leaky_brelu(net, self.conv_leaky_ratio[layer_id], self.layer_low_bound, self.output_up_bound) # Nonlinear act
            if layer_id != len(self.out_fc_states) - 1: # last layer
                net = ne.leaky_relu(net, self.out_leaky_ratio)

        net = tf.identity(net, name='out_output')
        #import pdb; pdb.set_trace()
        return net


    def evaluate(self, data, is_training, use_summary=True):
        self.is_training = is_training
        conv_res = self.conv_res_groups(data, conv_residual=self.conv_residual)
        #assert res.get_shape().as_list()[1:] == self.res_out_shape
        if self.include_top:
            logits = self.out_layer(conv_res)
            preds = tf.nn.softmax(logits)
            # self.saver = tf.train.Saver()
            '''
            if use_summary:
                # tensorboard
                tf.summary.histogram("Logits", logits)
                tf.summary.histogram("Predictions", preds)
            '''
            return logits, preds
        else:
            return conv_res


    def prediction(self, data, use_summary=True):
        return self.evaluate(data, False, use_summary)


    def loss(self, logits, groundtruth, loss_type="xentropy"):
        if loss_type == "xentropy":
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=groundtruth, logits=logits
            ))
        elif loss_type == "mse":
            loss = tf.reduce_mean(
                tf.reduce_sum((tf.nn.softmax(logits) - groundtruth) ** 2, 1),
                0)
        # tensorboard
        #tf.summary.scalar("Loss", loss)
        return loss



    def optimization(self, learning_rate, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer


    def accuracy(self, prediction, groundtruth):
        correct_prediction = tf.equal(
            tf.argmax(groundtruth, 1),
            tf.argmax(prediction, 1)
        )
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tensorboard
        #tf.summary.scalar("Accuracy", acc)
        return acc


    def tf_load(self, sess, path, scope, name='mnist_cnn.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.restore(sess, path+'/'+name+spec)

    def tf_save(self, sess, path, scope, name='mnist_cnn.ckpt', spec=""):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver.save(sess, path+'/'+name+spec)
