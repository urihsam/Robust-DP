import nn.mnist_classifier as mnist_cnn
import utils.net_element as ne
from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer_v2 as dp_optimizer
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients
from differential_privacy.dp_sgd.dp_optimizer import utils
from utils.decorator import *
from dependency import *
import tensornets as nets
import os, math


class RDPCNN:
    """
    Robust Differential Privacte CNN
    """

    def __init__(self, data, label, input_sigma, is_training, noised_pre=None, noise=None):
        self.label = label
        self.data = data
        self.input_sigma = input_sigma
        self.is_training = is_training
        #
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.experimental.list_logical_devices('GPU')
        #with tf.device(gpus[0].name):

        # cnn model
        with tf.variable_scope('CNN') as scope:
            self.cnn = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [3,3], [4,4], [3,3], [4,4]],
                        conv_strides = [[2,2], [1,1], [2,2], [1,1], [2,2]], 
                        conv_channel_sizes=[8, 8, 64, 64, 16], 
                        conv_leaky_ratio=[0.2, 0.2, 0.2, 0.2, 0.2],
                        conv_drop_rate=[0.0, 0.2, 0.2,  0.2, 0.0],
                        conv_residual=True,
                        num_res_block=1,
                        res_block_size=1,
                        #num_res_block=0,
                        out_norm="NONE",
                        use_norm="NONE",
                        img_channel=1,
                        include_top=False)
            conv = self.cnn.evaluate(self.data, self.is_training)
        self.pre_conv = tf.reshape(conv, [conv.get_shape().as_list()[0], -1])

        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        # finetune
        with tf.variable_scope('top_layers') as opt_scope_1:
            w_2 = tf.get_variable(initializer=initializer, shape=(self.pre_conv.get_shape().as_list()[-1], 16), name="w_2")
            b_2 = tf.get_variable(initializer=initializer, shape=(16), name="b_2")
            w_3 = tf.get_variable(initializer=initializer, shape=(16,10), name="w_3")
            b_3 = tf.get_variable(initializer=initializer, shape=(10), name="b_3")
            
            net = self.pre_conv
            #net = ne.layer_norm(self.noised_pretrain, self.is_training)
            net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
            #net = ne.layer_norm(net, self.is_training)

            self.clean_logits = tf.add(tf.matmul(net, w_3), b_3)
            #self.cnn_logits = tf.add(tf.matmul(net, w_2), b_2)
            self.clean_prediction = tf.nn.softmax(self.clean_logits)
            
            correct_pred = tf.equal(tf.argmax(self.clean_prediction, 1), tf.argmax(self.label, 1))
            self.clean_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
        self.opt_scope_1 = opt_scope_1
        
        self.noised_pre = noised_pre
        self.pre = self.noised_pre - noise
        
        # finetune
        with tf.variable_scope(opt_scope_1, reuse=True):
            w_2 = tf.get_variable(initializer=initializer, shape=(self.noised_pre.get_shape().as_list()[-1], 16), name="w_2")
            b_2 = tf.get_variable(initializer=initializer, shape=(16), name="b_2")
            w_3 = tf.get_variable(initializer=initializer, shape=(16,10), name="w_3")
            b_3 = tf.get_variable(initializer=initializer, shape=(10), name="b_3")
            
            net = self.noised_pre
            #net = ne.layer_norm(self.noised_pretrain, self.is_training)
            net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
            #net = ne.layer_norm(net, self.is_training)

            self.finetune_logits = tf.add(tf.matmul(net, w_3), b_3)
            #self.cnn_logits = tf.add(tf.matmul(net, w_2), b_2)
            self.finetune_prediction = tf.nn.softmax(self.finetune_logits)
            
            correct_pred = tf.equal(tf.argmax(self.finetune_prediction, 1), tf.argmax(self.label, 1))
            self.finetune_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


        # finetune recon
        with tf.variable_scope(opt_scope_1, reuse=True):
            w_2 = tf.get_variable(initializer=initializer, shape=(self.pre.get_shape().as_list()[-1], 16), name="w_2")
            b_2 = tf.get_variable(initializer=initializer, shape=(16), name="b_2")
            w_3 = tf.get_variable(initializer=initializer, shape=(16,10), name="w_3")
            b_3 = tf.get_variable(initializer=initializer, shape=(10), name="b_3")
            
            net = self.pre
            #net = ne.layer_norm(self.pretrain, self.is_training)
            net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
            #net = ne.layer_norm(net, self.is_training)

            self.finetune_clean_logits = tf.add(tf.matmul(net, w_3), b_3)
            #self.cnn_clean_logits = tf.add(tf.matmul(net, w_2), b_2)
            self.finetune_clean_prediction = tf.nn.softmax(self.finetune_clean_logits)
            
            correct_pred = tf.equal(tf.argmax(self.finetune_clean_prediction, 1), tf.argmax(self.label, 1))
            self.finetune_clean_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    def get_label(self, prediction):
        n_class = prediction.get_shape().as_list()[1]
        indices = tf.argmax(prediction, axis=1)
        return tf.one_hot(indices, n_class, on_value=1.0, off_value=0.0)


    def vectorize(self, x):
        return tf.reshape(x, [-1, FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS * FLAGS.NUM_CHANNELS])
    

    @lazy_method
    def loss(self, coef, logits):
        loss = coef * tf.losses.softmax_cross_entropy(self.label, logits)
        return loss
    
    
    @lazy_method_no_scope
    def compute_M_from_input_perturbation(self, loss, l2norm_bound, is_layerwised=False, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            ex = self.noised_pre
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            #import pdb; pdb.set_trace()
            xs = [tf.convert_to_tensor(x) for x in var_list]
            #import pdb; pdb.set_trace()
            # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
            px_grads = per_example_gradients.PerExampleGradients(loss, xs)
            # calculate sigma, sigma has the shape of [batch_size]

            # layer-wised
            if is_layerwised:
                Ms = []
                sens = []
                for px_grad, v in zip(px_grads, var_list):
                    px_grad_vec = tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) # [batch_size, vec_param]
                    # Clipping
                    #px_grad_vec = utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                    
                    px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                    px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                    #
                    M = tf.reduce_mean(tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])), axis=0)/tf.shape(px_grad)[0] #[b, vec_param, vec_param]
                    #M = tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])) #[b, vec_param, vec_param]
                    
                    
                    Ms.append(M)
                    sens.append(px_grad_vec)
                #S_mins = tf.stack(S_mins, axis=1)
                return Ms, sens#, kk_square, mark_off, r_k, c_k, core, mask_on)
            else:
                # all in
                px_grad_vec_list = [tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) for px_grad in px_grads] # [batch_size, vec_param * L]
                px_grad_vec = tf.concat(px_grad_vec_list, axis=1) # [batch_size, vec_param]
                # Clipping
                #px_grad_vec = utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                #px_pp_grad2 = batch_jacobian(px_grad_vec, self.data, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                #px_pp_jac2 = tf.reshape(px_pp_grad2, [px_pp_grad2.get_shape().as_list()[0], px_pp_grad2.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                #
                M = tf.reduce_mean(tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])), axis=0)/tf.cast(tf.shape(px_grad_vec)[0], dtype=tf.float32) #[b, vec_param, vec_param]
                #M = tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])) #[b, vec_param, vec_param]

                #b_left = tf.linalg.lstsq(px_pp_jac, tf.eye(px_pp_grad.get_shape().as_list()[1], batch_shape=[px_pp_grad.get_shape().as_list()[0]]))
                
                return M, px_grad_vec

    
    def vars_size(self):
        return len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name))
    

    @lazy_method_no_scope
    def dp_optimization(self, loss, sanitizer, dp_sigma, trans_sigma=None, is_finetune=False, batched_per_lot=1, is_sigma_data_dependent=False, is_layerwised=False, scope="DP_OPT"):
        '''
        def no_dp():
            op, _, _, learning_rate = self.optimization(loss)
            return op, learning_rate
        
        def dp():
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

                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

                
                # dp opt
                optimizer = dp_optimizer.DPGradientDescentOptimizer(
                    learning_rate,
                    [None, None],
                    sanitizer,
                    sigma=dp_sigma,
                    batches_per_lot=batched_per_lot)
                
                # this is the dp minimization
                #import pdb; pdb.set_trace()
                #grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)
                #op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                # global_step is incremented by one after the variables have been updated.
                op = optimizer.minimize(loss, global_step=global_step, var_list=opt_vars)
                tf.summary.scalar("Learning_rate", learning_rate)
                return op, learning_rate
        return tf.cond(tf.equal(tf.reduce_sum(dp_sigma), tf.constant(0.0)), true_fn=no_dp, false_fn=dp)
        '''
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
            if is_finetune:
                learning_rate = tf.train.exponential_decay(FLAGS.FINETUNE_LEARNING_RATE, global_step, 
                    FLAGS.FINETUNE_LEARNING_DECAY_STEPS, FLAGS.FINETUNE_LEARNING_DECAY_RATE)
            else:
                learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step, 
                    FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)

            if is_finetune:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            else:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            
            # dp opt
            optimizer = dp_optimizer.DPGradientDescentOptimizer(
                learning_rate,
                [None, None],
                sanitizer,
                sigma=dp_sigma,
                accountant_sigma=trans_sigma,
                batches_per_lot=batched_per_lot,
                is_sigma_layerwised=is_layerwised,
                is_sigma_data_dependent=is_sigma_data_dependent)
            
            # this is the dp minimization
            #import pdb; pdb.set_trace()
            #grads_and_vars = optimizer.compute_gradients(loss, var_list=opt_vars)
            #op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # global_step is incremented by one after the variables have been updated.
            op = optimizer.minimize(loss, global_step=global_step, var_list=opt_vars)
            tf.summary.scalar("Learning_rate", learning_rate)
            return op, learning_rate
    
    
    @lazy_method_no_scope
    def optimization(self, loss, is_finetune=False, accum_iters=1, scope="OPT"):
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

            if is_finetune:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            else:
                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


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

            # tensorboard
            for g, v in grads_and_vars:
                # print v.name
                name = v.name.replace(":", "_")
                tf.summary.histogram(name+"_gradients", g)
            tf.summary.scalar("Learning_rate", learning_rate)
            return op, (zero_op, accum_op, avg_op), reset_decay_op, learning_rate
    

    def tf_load_pretrained(self, sess, scope="CNN", name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list)
        path = FLAGS.PRETRAINED_CNN_PATH
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))


    def tf_load(self, sess, scope=None, name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        if scope == None:
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="CNN") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
        else:
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list)
        path = FLAGS.CNN_PATH
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))


    def tf_save(self, sess, scope=None, name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        if scope == None:
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="CNN") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

        else:
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list)
        path = FLAGS.CNN_PATH
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(sess, path +'/'+name)
        print("Save model to {}".format(path +'/'+name))
