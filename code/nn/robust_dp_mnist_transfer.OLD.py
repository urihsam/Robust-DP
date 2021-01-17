import nn.mnist_classifier as mnist_cnn
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

    def __init__(self, label, input_sigma, is_training, data=None, noised_data=None, px_noised_data=None, noise=None):
        self.label = label
        self.input_sigma = input_sigma
        self.is_training = is_training
        #
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.experimental.list_logical_devices('GPU')
        #with tf.device(gpus[0].name): 
        if px_noised_data != None and noise != None:
            self.px_noised_data = px_noised_data
            self.noised_data = tf.concat(self.px_noised_data, axis=0)
            self.data = self.noised_data - noise
        elif noised_data != None and noise != None:
            self.noised_data = noised_data
            self.data = self.noised_data - noise
        else:
            self.data = data
            # noise layer
            self.noised_data = self.noise_layer(self.data, self.input_sigma)

        tiled_data = tf.tile(self.data, [1,1,1,3])
        tiled_noised_data = tf.tile(self.noised_data, [1,1,1,3])
        # cnn model
        with tf.variable_scope('CNN') as scope:
            pre_trained = nets.VGG19(tiled_noised_data, is_training=False, early_stem=True, stem=True)
            net =  tf.identity(pre_trained)
            print(net.get_shape().as_list())
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            
            w_1 = tf.get_variable(initializer=initializer, shape=(3, 3, net.get_shape().as_list()[-1], 128), name="w_1")
            b_1 = tf.get_variable(initializer=initializer, shape=(128), name="b_1")
            net = tf.nn.conv2d(net, w_1, strides=[1, 1, 1, 1], padding="SAME")+ b_1
            net = tf.nn.leaky_relu(net)
            net = tf.reshape(net, [FLAGS.BATCH_SIZE, -1])
        
        with tf.variable_scope('top_layers') as opt_scope_1:
            w_2 = tf.get_variable(initializer=initializer, shape=(net.get_shape().as_list()[-1], 16), name="w_2")
            b_2 = tf.get_variable(initializer=initializer, shape=(16), name="b_2")
            w_3 = tf.get_variable(initializer=initializer, shape=(16,10), name="w_3")
            b_3 = tf.get_variable(initializer=initializer, shape=(10), name="b_3")
            
            net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
            self.cnn_logits = tf.add(tf.matmul(net, w_3), b_3)
            #self.cnn_logits = tf.add(tf.matmul(net, w_2), b_2)
            self.cnn_prediction = tf.nn.softmax(self.cnn_logits)
            
            correct_pred = tf.equal(tf.argmax(self.cnn_prediction, 1), tf.argmax(self.label, 1))
            self.cnn_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.opt_scope_1 = opt_scope_1
        # recon
        with tf.variable_scope(scope, reuse=True):
            pre_trained = nets.VGG19(tiled_data, is_training=False, early_stem=True, stem=True)
            net =  tf.identity(pre_trained)
            print(net.get_shape().as_list())
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            
            w_1 = tf.get_variable(initializer=initializer, shape=(3, 3, net.get_shape().as_list()[-1], 128), name="w_1")
            b_1 = tf.get_variable(initializer=initializer, shape=(128), name="b_1")
            net = tf.nn.conv2d(net, w_1, strides=[1, 1, 1, 1], padding="SAME")+ b_1
            net = tf.nn.leaky_relu(net)
            net = tf.reshape(net, [FLAGS.BATCH_SIZE, -1])
        with tf.variable_scope(opt_scope_1, reuse=True):
            w_2 = tf.get_variable(initializer=initializer, shape=(net.get_shape().as_list()[-1], 16), name="w_2")
            b_2 = tf.get_variable(initializer=initializer, shape=(16), name="b_2")
            w_3 = tf.get_variable(initializer=initializer, shape=(16,10), name="w_3")
            b_3 = tf.get_variable(initializer=initializer, shape=(10), name="b_3")
            
            net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
            self.cnn_clean_logits = tf.add(tf.matmul(net, w_3), b_3)
            #self.cnn_clean_logits = tf.add(tf.matmul(net, w_2), b_2)
            self.cnn_clean_prediction = tf.nn.softmax(self.cnn_clean_logits)
            
            correct_pred = tf.equal(tf.argmax(self.cnn_clean_prediction, 1), tf.argmax(self.label, 1))
            self.cnn_clean_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    def noise_layer(self, data, sigma):
        noise_distribution = tf.distributions.Normal(loc=0.0, scale=sigma)
        noised_data = data + noise_distribution.sample(tf.shape(data))
        return noised_data


    def get_label(self, prediction):
        n_class = prediction.get_shape().as_list()[1]
        indices = tf.argmax(prediction, axis=1)
        return tf.one_hot(indices, n_class, on_value=1.0, off_value=0.0)


    def vectorize(self, x):
        return tf.reshape(x, [-1, FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS * FLAGS.NUM_CHANNELS])
    

    @lazy_method
    def loss(self):
        loss = FLAGS.BETA * tf.losses.softmax_cross_entropy(self.label, self.cnn_logits)
        tf.summary.scalar("Total_loss", loss)
        return loss


    @lazy_method
    def loss_clean(self):
        loss = FLAGS.BETA * tf.losses.softmax_cross_entropy(self.label, self.cnn_clean_logits)
        tf.summary.scalar("Total_loss_clean", loss)
        return loss


    @lazy_method_no_scope
    def compute_lower_bound_of_singular_value_1(self, M, param_size, scope="S_LOWER_BOUND_1"):
        with tf.variable_scope(scope):
            kk_square = tf.square(tf.linalg.diag_part(M)) #[b, vec_param]
            mark_off = tf.linalg.diag(tf.linalg.diag_part(M)) #[b, vec_param, vec_param]
            r_k = tf.reduce_sum(tf.abs(M-mark_off), axis=2) #[b, vec_param]
            c_k = tf.reduce_sum(tf.abs(M-mark_off), axis=1) #[b, vec_param]

            core = (tf.sqrt(4 * kk_square + tf.square(r_k - c_k)) - (r_k + c_k)) / 2
            mask_on = tf.cast(tf.math.greater_equal(core, 0), dtype=tf.float32)
            batch_S_min = tf.reduce_min(core*mask_on, axis=1) # [b]
            batch_S_min = tf.sqrt(batch_S_min / tf.cast(param_size, dtype=tf.float32))
            return batch_S_min


    @lazy_method_no_scope
    def compute_lower_bound_of_singular_value_2(self, M, param_size, scope="S_LOWER_BOUND_2"):
        with tf.variable_scope(scope):
            kk = tf.abs(tf.linalg.diag_part(M)) #[b, vec_param]
            mark_off = tf.linalg.diag(tf.linalg.diag_part(M)) #[b, vec_param, vec_param]
            r_k = tf.reduce_sum(tf.abs(M-mark_off), axis=2) #[b, vec_param]
            c_k = tf.reduce_sum(tf.abs(M-mark_off), axis=1) #[b, vec_param]
            core = kk - (r_k + c_k) /2.0
            mask_on = tf.cast(tf.math.greater_equal(core, 0), dtype=tf.float32)
            batch_S_min = tf.reduce_min(core*mask_on, axis=1) # [b]
            batch_S_min = tf.sqrt(batch_S_min / tf.cast(param_size, dtype=tf.float32))
            return batch_S_min


    @lazy_method_no_scope
    def compute_lower_bound_of_singular_value_3(self, M, param_size, scope="S_LOWER_BOUND_3"):
        with tf.variable_scope(scope):
            det_M = tf.identity(tf.linalg.det(M)) # [b]
            M_fro = tf.identity(tf.linalg.norm(M, ord="fro", axis=(1, 2))) # [b]
            l_base = tf.identity(tf.cast(param_size-1, dtype=tf.float32)/tf.square(M_fro))
            l = tf.identity(tf.abs(det_M) * tf.pow(l_base, tf.cast((param_size-1)/2, dtype=tf.float32)))
            S_base = tf.identity(tf.cast(param_size-1, dtype=tf.float32)/(tf.square(M_fro)-tf.square(l)))
            batch_S_min = tf.identity(tf.abs(det_M) * tf.pow(S_base, tf.cast((param_size-1)/2, dtype=tf.float32)))
            batch_S_min = tf.sqrt(batch_S_min / tf.cast(param_size, dtype=tf.float32))
            return batch_S_min


    @lazy_method_no_scope
    def compute_lower_bound_of_singular_value_4(self, M, param_size, scope="S_LOWER_BOUND_4"):
        with tf.variable_scope(scope):
            det_M = tf.identity(tf.linalg.det(M)) # [b]
            base = tf.identity(tf.cast((param_size-1)/param_size, dtype=tf.float32))
            batch_S_min = tf.identity(tf.abs(det_M) * tf.pow(base, tf.cast((param_size-1)/2, dtype=tf.float32)))
            batch_S_min = tf.sqrt(batch_S_min / tf.cast(param_size, dtype=tf.float32))
            return batch_S_min
    

    @lazy_method_no_scope
    def compute_S_min_from_input_perturbation(self, loss, is_layerwised=False, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            ex = self.noised_data
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            #import pdb; pdb.set_trace()
            xs = [tf.convert_to_tensor(x) for x in var_list]
            #import pdb; pdb.set_trace()
            # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
            px_grads = per_example_gradients.PerExampleGradients(loss, xs)
            # calculate sigma, sigma has the shape of [batch_size]
            
            # layer-wised
            if is_layerwised:
                S_mins = []
                for px_grad, v in zip(px_grads, var_list):
                    px_grad_vec = tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) # [batch_size, vec_param]
                    px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                    px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                    #
                    M = tf.identity(tf.matmul(tf.identity(px_pp_jac), tf.identity(tf.transpose(px_pp_jac, [0, 2, 1])))) #[b, vec_param, vec_param]
                    #
                    param_size = tf.identity(px_pp_grad.get_shape().as_list()[1])
                    # method1
                    batch_S_min = tf.identity(self.compute_lower_bound_of_singular_value_4(M, param_size))
                    
                    S_mins.append(batch_S_min)
                #S_mins = tf.stack(S_mins, axis=1)
                return S_mins, (M)#, kk_square, mark_off, r_k, c_k, core, mask_on)
            else:
                # all in
                px_grad_vec_list = [tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) for px_grad in px_grads] # [batch_size, vec_param * L]
                px_grad_vec = tf.concat(px_grad_vec_list, axis=1) # [batch_size, vec_param]
                px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                #
                M = tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])) #[b, vec_param, vec_param]
                #
                param_size = px_pp_grad.get_shape().as_list()[1]
                # method1
                batch_S_min = self.compute_lower_bound_of_singular_value_4(M, param_size)

                S_min = tf.reduce_min(batch_S_min)
                return S_min, (M)

    
    @lazy_method_no_scope
    def compute_Jac_from_input_perturbation(self, loss, l2norm_bound, is_layerwised=False, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            ex = self.noised_data
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            #import pdb; pdb.set_trace()
            xs = [tf.convert_to_tensor(x) for x in var_list]
            #import pdb; pdb.set_trace()
            # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
            px_grads = per_example_gradients.PerExampleGradients(loss, xs)
            # calculate sigma, sigma has the shape of [batch_size]

            # layer-wised
            if is_layerwised:
                Jacs = []
                sens = []
                for px_grad, v in zip(px_grads, var_list):
                    px_grad_vec = tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) # [batch_size, vec_param]
                    # Clipping
                    #px_grad_vec = utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                    
                    px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                    px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                    #
                    M = tf.reduce_mean(tf.matmul(px_pp_jac, tf.transpose(px_pp_jac, [0, 2, 1])), axis=0)/tf.shape(px_grad)[0] #[b, vec_param, vec_param]
                    
                    
                    Jacs.append(M)
                    sens.append(px_grad_vec)
                #S_mins = tf.stack(S_mins, axis=1)
                return Jacs, sens#, kk_square, mark_off, r_k, c_k, core, mask_on)
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
                #b_left = tf.linalg.lstsq(px_pp_jac, tf.eye(px_pp_grad.get_shape().as_list()[1], batch_shape=[px_pp_grad.get_shape().as_list()[0]]))
                
                return M, px_grad_vec

    
    @lazy_method_no_scope
    def compute_Jac_from_input_perturbation_2(self, loss, l2norm_bound, is_layerwised=False, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            ex = self.noised_data
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
            #import pdb; pdb.set_trace()
            params = [tf.convert_to_tensor(x) for x in var_list]
            #import pdb; pdb.set_trace()
            # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
            #px_grads = per_example_gradients.PerExampleGradients(loss, [ex])
            px_grads = tf.gradients(loss, ex)[0]
            #px_grads = tf.concat(px_grads, axis=0)
            px_grads_vec = tf.reshape(px_grads, [px_grads.get_shape().as_list()[0], -1]) # [batch_size, vec_param]
            # calculate sigma, sigma has the shape of [batch_size]
            #import pdb; pdb.set_trace()
            Jacs = []
            for param, v in zip(params, var_list):
                # Clipping
                #px_grad_vec = utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
                
                px_pp_grad = jacobian(px_grads_vec, param, use_pfor=False, parallel_iterations=px_grads_vec.get_shape().as_list()[0]*px_grads_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
                px_pp_jac = tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1]) #[b, vec_param, ex_size]
                #
                #M = tf.identity(tf.matmul(tf.identity(px_pp_jac), tf.identity(tf.transpose(px_pp_jac, [0, 2, 1])))) #[b, vec_param, vec_param]
                
                px_pp_jac = tf.transpose(px_pp_jac, [0, 2, 1])
                Jacs.append(px_pp_jac)
            # layer-wised
            if is_layerwised:
                
                #S_mins = tf.stack(S_mins, axis=1)
                return Jacs, Jacs
            else:
                px_pp_jac = tf.concat(Jacs, axis=1)
                return px_pp_jac, px_pp_jac

    
    def vars_size(self):
        return len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name))
    

    @lazy_method_no_scope
    def dp_optimization(self, loss, sanitizer, dp_sigma, trans_sigma, batched_per_lot=1, is_sigma_data_dependent=False, is_layerwised=False, scope="DP_OPT"):
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
            learning_rate = tf.train.exponential_decay(FLAGS.LEARNING_RATE, global_step, 
                FLAGS.LEARNING_DECAY_STEPS, FLAGS.LEARNING_DECAY_RATE)

            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

            
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
    def optimization(self, loss, accum_iters=1, scope="OPT"):
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
