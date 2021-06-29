from utils.fgsm_attack import fgm
import nn.mnist_classifier as mnist_cnn
import utils.net_element as ne
from tensornets.preprocess import keras_resnet_preprocess
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients
from differential_privacy.dp_sgd.dp_optimizer import utils as dp_utils
import nn.ae.cenc as cenc
import nn.ae.cdec as cdec
from utils.decorator import *
from dependency import *
import os, math


class DP_DENOISER:
    """
    Sliced Network
    """

    def __init__(self, data, label, is_training, noised_data=None, data_infer=None, recon_infer=None):
        self.data = data
        self.label = label
        self.is_training = is_training
        self.noised_data = noised_data
        self.data_infer = data_infer
        self.recon_infer = recon_infer
        #
        self.batch_size = self.data.get_shape().as_list()[0]
        #
        self.output_low_bound = 0.0
        self.output_up_bound = 1.0




    def reconstruct(self, data, reuse_param=False):
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True) # Convolutional
        initializer2 = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
        drop_rate = FLAGS.DROP_RATE # 0.3
        with tf.variable_scope("DP_DENOISER", reuse=reuse_param):
            with tf.variable_scope('AUTOENCODER', reuse=reuse_param) as ae_scope:
                # Encoder
                with tf.variable_scope("BOTT_ENC", reuse=reuse_param) as enc_bott_scope:
                    w_0 = tf.get_variable(initializer=initializer, shape=(4, 4, 1, 32), name="W_conv_0",)
                    b_0 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(32), name="b_conv_0")

                    conv_in = data

                    #net = ne.layer_norm(net, self.is_training)
                    net = tf.nn.conv2d(conv_in, w_0, strides=[1, 2, 2, 1], padding="SAME")+ b_0
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    conv_0 = ne.layer_norm(net, self.is_training)

                self.enc_bott_scope = enc_bott_scope

                self.enc_in = conv_0
                with tf.variable_scope("ENC", reuse=reuse_param) as enc_scope:
                    w_1 = tf.get_variable(initializer=initializer, shape=(3, 3, 32, 32), name="W_conv_1")
                    b_1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(32), name="b_conv_1")
                    w_2 = tf.get_variable(initializer=initializer, shape=(4, 4, 32, 128), name="W_conv_2")
                    b_2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_2")
                    #w_3 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_conv_3")
                    #b_3 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_3")
                    w_4 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 128), name="W_conv_4")
                    b_4 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_4")
                    #w_5 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_conv_5")
                    #b_5 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_5")
                    w_6 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 128), name="W_conv_6")
                    b_6 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_6")
                    #w_7 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_conv_7")
                    #b_7 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_conv_7")
                    #w_8 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 512), name="W_conv_8")
                    #b_8 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(512), name="b_conv_8")

                    res = conv_0
                    net = tf.nn.conv2d(conv_0, w_1, strides=[1, 1, 1, 1], padding="SAME")+ b_1
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    conv_1 = ne.layer_norm(net, self.is_training)

                    net = tf.nn.conv2d(conv_1, w_2, strides=[1, 2, 2, 1], padding="SAME")+ b_2
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    conv_2 = ne.layer_norm(net, self.is_training)

                    '''
                    res = conv_2
                    net = tf.nn.conv2d(conv_2, w_3, strides=[1, 1, 1, 1], padding="SAME")+ b_3
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    conv_3 = ne.layer_norm(net, self.is_training)
                    '''

                    #net = tf.nn.conv2d(conv_3, w_4, strides=[1, 2, 2, 1], padding="SAME")+ b_4
                    net = tf.nn.conv2d(conv_2, w_4, strides=[1, 2, 2, 1], padding="SAME")+ b_4
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    conv_4 = ne.layer_norm(net, self.is_training)

                    '''
                    res = conv_4
                    net = tf.nn.conv2d(conv_4, w_5, strides=[1, 1, 1, 1], padding="SAME")+ b_5
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    conv_5 = ne.layer_norm(net, self.is_training)
                    '''


                    #net = tf.nn.conv2d(conv_5, w_6, strides=[1, 2, 2, 1], padding="SAME")+ b_6
                    net = tf.nn.conv2d(conv_4, w_6, strides=[1, 2, 2, 1], padding="SAME")+ b_6
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    conv_6 = ne.layer_norm(net, self.is_training)

                    '''
                    res = conv_6
                    net = tf.nn.conv2d(conv_6, w_7, strides=[1, 1, 1, 1], padding="SAME")+ b_7
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    conv_7 = ne.layer_norm(net, self.is_training)
                    '''

                    '''
                    #net = tf.nn.conv2d(conv_7, w_8, strides=[1, 2, 2, 1], padding="SAME")+ b_8
                    net = tf.nn.conv2d(conv_6, w_8, strides=[1, 2, 2, 1], padding="SAME")+ b_8
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    conv_8 = ne.layer_norm(net, self.is_training)
                    '''

                # Decoder
                with tf.variable_scope("DEC", reuse=reuse_param) as dec_scope:
                    #w_0 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 512), name="W_decv_0")
                    #b_0 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_0")
                    #w_1 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_decv_1")
                    #b_1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_1")
                    w_2 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 128), name="W_decv_2")
                    b_2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_2")
                    #w_3 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_decv_3")
                    #b_3 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_3")
                    w_4 = tf.get_variable(initializer=initializer, shape=(4, 4, 128, 128), name="W_decv_4")
                    b_4 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_4")
                    #w_5 = tf.get_variable(initializer=initializer, shape=(3, 3, 128, 128), name="W_decv_5")
                    #b_5 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(128), name="b_decv_5")
                    w_6 = tf.get_variable(initializer=initializer, shape=(4, 4, 32, 128), name="W_decv_6")
                    b_6 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(32), name="b_decv_6")
                    w_7 = tf.get_variable(initializer=initializer, shape=(3, 3, 32, 32), name="W_decv_7")
                    b_7 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(32), name="b_decv_7")
                    #import pdb; pdb.set_trace()


                    '''
                    net = ne.conv2d_transpose(conv_8, w_0, b_0, strides=[2, 2], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    decv_0 = ne.layer_norm(net, self.is_training)
                    '''

                    '''
                    res = decv_0 + conv_4
                    net = ne.conv2d_transpose(decv_0, w_1, b_1, strides=[1, 1], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    decv_1 = ne.layer_norm(net, self.is_training)
                    '''


                    #net = ne.conv2d_transpose(decv_1, w_2, b_2, strides=[2, 2], padding="SAME")
                    net = ne.conv2d_transpose(conv_6, w_2, b_2, strides=[2, 2], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    decv_2 = ne.layer_norm(net, self.is_training)
                    '''
                    res = decv_2
                    net = ne.conv2d_transpose(decv_2, w_3, b_3, strides=[1, 1], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    decv_3 = ne.layer_norm(net, self.is_training)
                    '''

                    #net = ne.conv2d_transpose(decv_3, w_4, b_4, strides=[2, 2], padding="SAME")
                    net = ne.conv2d_transpose(decv_2, w_4, b_4, strides=[2, 2], padding="SAME", odd=True)
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    decv_4 = ne.layer_norm(net, self.is_training)

                    '''
                    res = decv_4 + conv_2
                    net = ne.conv2d_transpose(decv_4, w_5, b_5, strides=[1, 1], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    decv_5 = ne.layer_norm(net, self.is_training)
                    '''

                    #net = ne.conv2d_transpose(decv_5, w_6, b_6, strides=[2, 2], padding="SAME")
                    net = ne.conv2d_transpose(decv_4, w_6, b_6, strides=[2, 2], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    decv_6 = ne.layer_norm(net, self.is_training)

                    res = decv_6 + conv_1
                    net = ne.conv2d_transpose(decv_6, w_7, b_7, strides=[1, 1], padding="SAME")
                    net = tf.nn.leaky_relu(net)
                    net = ne.drop_out(net, drop_rate, self.is_training)
                    net += res
                    decv_7 = ne.layer_norm(net, self.is_training)


                self.dec_out = decv_7

                self.enc_scope = enc_scope
                self.dec_scope = dec_scope

                with tf.variable_scope("TOP_DEC", reuse=reuse_param) as dec_top_scope:
                    w_8 = tf.get_variable(initializer=initializer, shape=(4, 4, 1, 32), name="W_dec_8")
                    b_8 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), name="b_dec_8")

                    net = decv_7 #+ conv_0 + conv_1
                    net = ne.conv2d_transpose(net, w_8, b_8, strides=[2, 2], padding="SAME")
                    #generated = ne.brelu(net, self.output_low_bound, self.output_up_bound)
                    #generated = net
                    generated = ne.sigmoid(net)

                self.dec_top_scope = dec_top_scope

        return generated

    
    def prediction(self, data, reuse_param=False):
        
        with tf.variable_scope('mnistcnn', reuse=reuse_param) as mnistcnn_scope:
            cnn = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                            conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                            conv_channel_sizes=[8, 8, 128, 128, 256], 
                            conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                            conv_drop_rate=[0.0, 0.2, 0.2, 0.2, 0.0],
                            conv_residual=True,
                            num_res_block=1,
                            res_block_size=1,
                            #num_res_block=0,
                            out_state=4*4*256,
                            out_fc_states=[1024, 10],
                            out_leaky_ratio=0.2,
                            out_norm="NONE",
                            use_norm="NONE",
                            img_channel=1)
            logits, prediction = cnn.prediction(data)
            accuracy = cnn.accuracy(prediction, self.label)
    
        return logits, prediction, accuracy, mnistcnn_scope


    def __call__(self, noised_data, data, reuse_param=False):
        self.noised_data = noised_data
        self.data = data
        self.recon_data = self.reconstruct(self.noised_data, reuse_param)
        self.recon_clean_data = self.reconstruct(self.data, True)

        self.data_logits, self.data_prediction, self.clean_accuracy, self.MNISTcnn_scope = self.prediction(self.data)
        self.recon_logits, self.recon_prediction, self.recon_accuracy, _ = self.prediction(self.recon_data, True)
        _, _, self.recon_clean_accuracy, _ = self.prediction(self.recon_clean_data, True)
        return self.recon_data, (self.recon_logits, self.recon_prediction), (self.data_logits, self.data_prediction)


    def get_label(self, prediction):
        n_class = prediction.get_shape().as_list()[1]
        indices = tf.argmax(prediction, axis=1)
        return tf.one_hot(indices, n_class, on_value=1.0, off_value=0.0)


    def vectorize(self, x):
        shape = x.get_shape().as_list()
        return tf.reshape(x, [shape[0], -1])




    @lazy_method
    def loss(self, coef):
        in_ = self.vectorize(self.data)
        out_ = self.vectorize(self.recon_data)
        loss = coef*tf.reduce_mean(tf.reduce_sum(tf.square(in_ - out_), [1]))

        #loss += coef * -1 * tf.reduce_mean(tf.reduce_sum(in_ * tf.log(tf.maximum(out_, 1e-3)) + (1-in_) * tf.log(tf.maximum((1-out_), 1e-3)), 1))
        
        '''
        enc_in_ = self.vectorize(self.enc_in)
        dec_out_ = self.vectorize(self.dec_out)
        loss += coef*tf.reduce_mean(tf.reduce_sum(tf.square(enc_in_ - dec_out_), [1]))
        '''
        '''
        loss += coef * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.label, logits=self.recon_logits
            ))
        '''
        return loss


    @lazy_method
    def loss_clean(self, coef):
        in_ = self.vectorize(self.data)
        out_ = self.vectorize(self.recon_clean_data)
        loss = coef*tf.reduce_mean(tf.reduce_sum(tf.square(in_ - out_), [1]))


        #loss += coef * -1 * tf.reduce_mean(tf.reduce_sum(in_ * tf.log(tf.maximum(out_, 1e-3)) + (1-in_) * tf.log(tf.maximum((1-out_), 1e-3)), 1))
        '''
        enc_in_ = self.vectorize(self.enc_in)
        dec_out_ = self.vectorize(self.dec_out)
        loss += coef*tf.reduce_mean(tf.reduce_sum(tf.square(enc_in_ - dec_out_), [1]))
        '''
        return loss


    @lazy_method
    def loss_reg(self, opt_vars=None, scope=None):
        if opt_vars == None:
            if scope==None:
                scope = "DP_DENOISER"
            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        regularize = tf.contrib.layers.l1_l2_regularizer(FLAGS.REG_SCALE, FLAGS.REG_SCALE)
        #print tf.GraphKeys.TRAINABLE_VARIABLES
        reg_term = sum([regularize(param) for param in opt_vars])

        return reg_term

    @lazy_method_no_scope
    def compute_M_from_input_perturbation(self, data, loss, l2norm_bound, var_list=None, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            ex = data
            if var_list == None:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

            xs = [tf.convert_to_tensor(x) for x in var_list]
            gradients = tf.gradients(loss, xs)
            #'''
            px_grads = []
            for grad in gradients:
                grad = tf.expand_dims(grad, axis=0)
                px_grads.append(tf.tile(grad, [self.batch_size]+[1]*(len(grad.get_shape())-1)))
            
            # calculate sigma, sigma has the shape of [batch_size]
            #import pdb; pdb.set_trace()
            # all in
            px_grad_vec_list = [tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) for px_grad in px_grads] # [batch_size, vec_param * L]
            px_grad_vec = tf.concat(px_grad_vec_list, axis=1) # [batch_size, vec_param]
            # Clipping
            px_grad_vec = dp_utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
            px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
            #px_pp_grad2 = batch_jacobian(px_grad_vec, self.data, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
            '''
            px_grads = gradients
            px_grad_vec_list = [tf.reshape(px_grad, [-1]) for px_grad in px_grads] # [vec_param * L]
            px_grad_vec = tf.concat(px_grad_vec_list, axis=0) # [ec_param]
            px_pp_grad = jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=self.batch_size)
            px_pp_grad = tf.transpose(px_pp_grad, [1, 0, 2, 3, 4])
            '''
            px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
            #px_pp_jac2 = tf.reshape(px_pp_grad2, [px_pp_grad2.get_shape().as_list()[0], px_pp_grad2.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
            #
            M = tf.reduce_mean(tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])), axis=0)/tf.cast(tf.shape(px_grad_vec)[0], dtype=tf.float32) #[b, vec_param, vec_param]
            #M = tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])) #[b, vec_param, vec_param]

            #b_left = tf.linalg.lstsq(px_pp_jac, tf.eye(px_pp_grad.get_shape().as_list()[1], batch_shape=[px_pp_grad.get_shape().as_list()[0]]))

            return M, px_grad_vec

        
    
    @lazy_method_no_scope
    def dp_optimization(self, loss, priv_accountant, dp_sigma, act_sigma=None, opt_vars=None, batches_per_lot=1, 
                        learning_rate=None, lr_decay_steps=None, lr_decay_rate=None, 
                        px_clipping_norm=1.0, scope="DP_OPT"):
        with tf.variable_scope(scope):
            """
            decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)
            learning rate decay with decay_rate per decay_steps
            """
            # use lobal step to keep track of our iterations
            global_step = tf.Variable(0, name="OPT_GLOBAL_STEP", trainable=False)

            # reset global step
            reset_decay_op = global_step.assign(tf.constant(0))
            # decay
            if learning_rate == None:
                learning_rate = FLAGS.LEARNING_RATE
            if lr_decay_steps == None:
                lr_decay_steps = FLAGS.LEARNING_DECAY_STEPS
            if lr_decay_rate == None:
                lr_decay_rate = FLAGS.LEARNING_DECAY_RATE
            learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                    lr_decay_steps, lr_decay_rate)

            if opt_vars == None:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

            momentum = 0.9
            """optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
                loss, var_list=opt_vars)"""
            if FLAGS.OPT_TYPE == "NEST":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            elif FLAGS.OPT_TYPE == "MOME":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=False)
            elif FLAGS.OPT_TYPE == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #dp opt
            grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)
            gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
            #import pdb; pdb.set_trace()
            #
            # act_sigma: input perturb is transformed into grad perturb, if transformed perturb is larger than grad perturb, 
            # then act sigma is transformed perturb and dp_sigma is 0, else act sigma is grad perturb, but dp_sigma is grad_perturb-act_sigma
            # DP accounting
            batch_size = self.batch_size
            lot_size = batch_size * batches_per_lot
            act_sigma_64 = tf.cast(act_sigma, tf.float64)
            privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], act_sigma_64, lot_size)
            
            # clipping
            clipped_grads = []
            for grad, var in zip(gradients, opt_vars):
                if len(var.shape) == 1 and len(grad.shape) == 3:
                    grad = tf.reduce_mean(grad, axis=[0,1])
                clipped_grads.append(tf.clip_by_norm(grad, px_clipping_norm))

            # accumulate
            if batches_per_lot != 1:
                accum_grads = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in clipped_grads]

                zero_op = [grads.assign(tf.zeros_like(grads)) for grads in accum_grads]
                accum_op = [accum_grads[i].assign_add(g) for i, g in enumerate(clipped_grads) if g!= None]
                avg_op = [grads.assign(grads/batches_per_lot) for grads in accum_grads]
            else:
                zero_op = None
                accum_op = None
                avg_op = None
                accum_grads = clipped_grads
    
            clipped_gradients = accum_grads
            # Add noise
            ratio = px_clipping_norm
            def noise():
                grads = []
                for grad in clipped_gradients:
                    with tf.control_dependencies([privacy_accum_op]):
                        grads.append(dp_utils.AddGaussianNoise(grad, dp_sigma * ratio))
                return grads
            def no_noise():
                grads = []
                for grad in clipped_gradients:
                    with tf.control_dependencies([privacy_accum_op]):
                        grads.append(grad)
                return grads
            
            gradients = tf.cond(tf.equal(dp_sigma, tf.constant(0.0)), no_noise, noise)
            grads_and_vars = zip(gradients, opt_vars)

            # global_step is incremented by one after the variables have been updated.
            op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            return op, (zero_op, accum_op, avg_op), reset_decay_op, learning_rate
    
    
    @lazy_method_no_scope
    def optimization(self, loss, opt_vars=None, accum_iters=1, scope="OPT"):
        with tf.variable_scope(scope):
            """
            decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)
            learning rate decay with decay_rate per decay_steps
            """
            # use lobal step to keep track of our iterations
            global_step = tf.Variable(0, name="OPT_GLOBAL_STEP", trainable=False)

            # reset global step
            reset_decay_op = global_step.assign(tf.constant(0))
            # decay
            learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step,
                FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)
            
            if opt_vars == None:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)


            momentum = 0.9
            """optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(
                loss, var_list=opt_vars)"""
            if FLAGS.OPT_TYPE == "NEST":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            elif FLAGS.OPT_TYPE == "MOME":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=False)
            elif FLAGS.OPT_TYPE == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)

            gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
            # accumulate
            if accum_iters != 1:
                accum_grads = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in variables]

                zero_op = [grads.assign(tf.zeros_like(grads)) for grads in accum_grads]
                accum_op = [accum_grads[i].assign_add(g) for i, g in enumerate(gradients) if g!= None]
                avg_op = [grads.assign(grads/accum_iters) for grads in accum_grads]
            else:
                zero_op = None
                accum_op = None
                avg_op = None
                accum_grads = gradients
            if FLAGS.IS_GRAD_CLIPPING:
                """
                https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
                clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
                t_list[i] * clip_norm / max(global_norm, clip_norm),
                where global_norm = sqrt(sum([l2norm(t)**2 for t in t_list])
                if clip_norm < global_norm, the gradients will be scaled to smaller values,
                especially, if clip_norm == 1, the graidents will be normed
                """
                clipped_grads, global_norm = (
                    #tf.clip_by_global_norm(gradients) )
                    tf.clip_by_global_norm(accum_grads, clip_norm=FLAGS.GRAD_CLIPPING_NORM))
                grads_and_vars = zip(clipped_grads, variables)
            else:
                grads_and_vars = zip(accum_grads, variables)
            # global_step is incremented by one after the variables have been updated.
            op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            return op, (zero_op, accum_op, avg_op), reset_decay_op, learning_rate


    def tf_load_classifier(self, sess, scope=None, name='sliced_ae.ckpt'):
        if scope == None:
            scope = self.MNISTcnn_scope.name
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = FLAGS.PRETRAINED_CNN_PATH
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))
    

    def tf_load(self, sess, scope=None, name='sliced_ae.ckpt'):
        if scope == None:
            scope = "DP_DENOISER"
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = "./models/MNIST/"+scope
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))

    def tf_save(self, sess, scope=None, name='sliced_ae.ckpt'):
        if scope == None:
            scope = "DP_DENOISER"
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = "./models/MNIST/"+scope
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(sess, path +'/'+name)
        print("Save model to {}".format(path +'/'+name))
