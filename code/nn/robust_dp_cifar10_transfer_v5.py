import nn.mnist_classifier as mnist_cnn
import utils.net_element as ne
from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer_v3 as dp_optimizer
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients
from differential_privacy.dp_sgd.dp_optimizer import utils as dp_utils
from utils.decorator import *
from dependency import *
import tensornets as nets
import os, math


class RDPCNN:
    """
    Robust Differential Privacte CNN
    """

    def __init__(self, data, label, input_sigma, is_training, noise=None):
        self.label = label
        self.data = data
        self.input_sigma = input_sigma
        self.is_training = is_training
        #
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.experimental.list_logical_devices('GPU')
        #with tf.device(gpus[0].name):

        # cnn model
        data_resized = tf.image.resize(self.data, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
        self.pre_trained = nets.VGG19(data_resized, is_training=True, stem=True)
        net_list = tf.split(self.pre_trained, 16, axis=-1)
        rows = []
        for i in range(4):
            rows.append(tf.concat(net_list[i*4:(i+1)*4], axis=1))
        net = tf.concat(rows, axis=2)
        self.pre_trained_cnn = net

        
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True) # conv
        initializer2 = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False) 
        with tf.variable_scope('top_layers_1') as opt_scope_1:
            w_0 = tf.get_variable(initializer=initializer, shape=(4, 4, 32, 2), name="W_conv_0")
            b_0 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(2), name="b_conv_0")
            w_1 = tf.get_variable(initializer=initializer, shape=(4, 4, 2, 2), name="W_conv_1")
            b_1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(2), name="b_conv_1")
            w_2 = tf.get_variable(initializer=initializer, shape=(4, 4, 2, 2), name="W_conv_2")
            b_2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(2), name="b_conv_2")

            w_4 = tf.get_variable(initializer=initializer2, shape=(32, 10), name="W_4")
            b_4 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(10), name="b_4")

            #import pdb; pdb.set_trace()
            net = self.pre_trained_cnn
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_0, strides=[1, 2, 2, 1], padding="SAME")+ b_0)
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_1, strides=[1, 2, 2, 1], padding="SAME")+ b_1)
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_2, strides=[1, 2, 2, 1], padding="SAME")+ b_2)
            net = tf.reshape(net, [self.pre_trained_cnn.get_shape().as_list()[0], -1])
            self.cnn_logits = tf.add(tf.matmul(net, w_4), b_4)
            self.cnn_prediction = tf.nn.softmax(self.cnn_logits)

            correct_pred = tf.equal(tf.argmax(self.cnn_prediction, 1), tf.argmax(self.label, 1))
            self.cnn_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.opt_scope_1 = opt_scope_1
        #self.opt_scope_2 = opt_scope_2

        # pub
        with tf.variable_scope(opt_scope_1, reuse=True): 
            net = self.pre_trained_cnn
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_0, strides=[1, 2, 2, 1], padding="SAME")+ b_0)
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_1, strides=[1, 2, 2, 1], padding="SAME")+ b_1)
            net = tf.nn.leaky_relu(tf.nn.conv2d(net, w_2, strides=[1, 2, 2, 1], padding="SAME")+ b_2)
            net = tf.reshape(net, [self.pre_trained_cnn.get_shape().as_list()[0], -1])

            self.cnn_pub_logits = tf.add(tf.matmul(net, w_4), b_4)
            self.cnn_pub_prediction = tf.nn.softmax(self.cnn_pub_logits)

            correct_pred = tf.equal(tf.argmax(self.cnn_pub_prediction, 1), tf.argmax(self.label, 1))
            self.cnn_pub_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


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
    def loss(self, beta):
        loss = beta* tf.losses.softmax_cross_entropy(self.label, self.cnn_logits)
        tf.summary.scalar("Total_loss", loss)
        return loss

    @lazy_method
    def loss_pub(self, beta):
        loss = beta * tf.losses.softmax_cross_entropy(self.label, self.cnn_pub_logits)
        tf.summary.scalar("Total_loss_pub", loss)
        return loss


    @lazy_method
    def loss_reg(self, reg_scale, opt_vars=None):
        if opt_vars == None:
            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)
        #print tf.GraphKeys.TRAINABLE_VARIABLES
        reg_term = reg_scale * sum([tf.reduce_sum(tf.square(param)) for param in opt_vars])
        #reg_term = FLAGS.REG_SCALE * sum([tf.reduce_sum(tf.math.abs(param)) for param in opt_vars])

        return reg_term

    @lazy_method_no_scope
    def compute_M_from_input_perturbation(self, loss, l2norm_bound, var_list=None, is_layerwised=False, scope="DP_S_MIN"):
        with tf.variable_scope(scope):
            #ex = self.noised_pre
            ex = self.data
            if var_list == None:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opt_scope_1.name)

            #import pdb; pdb.set_trace()
            xs = [tf.convert_to_tensor(x) for x in var_list]
            #import pdb; pdb.set_trace()
            # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
            px_grads = per_example_gradients.PerExampleGradients(loss, xs)
            '''
            px_grads_per_loss = []
            for loss in loss_list:
                px_grads_per_loss.append(per_example_gradients.PerExampleGradients(loss, xs))
            per_loss_px_grads = list(zip(*px_grads_per_loss))
            px_grads = []
            for grads in per_loss_px_grads:
                s_ = tf.constant(0.0)
                for g_ in grads:
                    s_ += g_
                px_grads.append(s_)
            '''
            # calculate sigma, sigma has the shape of [batch_size]
            #import pdb; pdb.set_trace()
            # layer-wised
            if is_layerwised:
                Ms = []
                sens = []
                for px_grad, v in zip(px_grads, var_list):
                    px_grad_vec = tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) # [batch_size, vec_param]
                    # Clipping
                    #px_grad_vec = dp_utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)

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
                #px_grad_vec = dp_utils.BatchClipByL2norm(px_grad_vec, FLAGS.DP_GRAD_CLIPPING_L2NORM)
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
    def dp_optimization(self, loss, priv_accountant, dp_sigma, act_sigma=None, opt_vars=None, batched_per_lot=1, 
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
            # px grads
            xs = [tf.convert_to_tensor(x) for x in opt_vars]
            px_grads = per_example_gradients.PerExampleGradients(loss, xs)
            #
            # act_sigma: input perturb is transformed into grad perturb, if transformed perturb is larger than grad perturb, 
            # then act sigma is transformed perturb and dp_sigma is 0, else act sigma is grad perturb, but dp_sigma is grad_perturb-act_sigma
            # DP accounting
            batch_size = px_grads[0].get_shape().as_list()[0]
            lot_size = batch_size * batched_per_lot
            act_sigma_64 = tf.cast(act_sigma, tf.float64)
            privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], act_sigma_64, lot_size)
            
            # clipping
            px_grad_vec_list = [tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) for px_grad in px_grads] # [batch_size, vec_param * L]
            px_grad_vec = tf.concat(px_grad_vec_list, axis=1) # [batch_size, vec_param]
            # clipping
            clipping_inv = 1.0/px_clipping_norm
            # Add a small number to avoid divide by 0
            l2norm_inv = tf.rsqrt(tf.reduce_sum(px_grad_vec * px_grad_vec, [1]) + 0.000001)
            scale = tf.minimum(l2norm_inv, clipping_inv) * px_clipping_norm
            
            clipped_px_grads = []
            for px_grad in px_grads:
                shape = px_grad.get_shape().as_list()
                px_grad_ = tf.reshape(px_grad, [shape[0], -1])
                scaled = tf.matmul(tf.diag(scale), px_grad_)
                clipped_px_grads.append(tf.reshape(scaled, shape))
            

            # accumulate
            if batched_per_lot != 1:
                accum_grads = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in clipped_px_grads]

                zero_op = [grads.assign(tf.zeros_like(grads)) for grads in accum_grads]
                accum_op = [accum_grads[i].assign_add(g) for i, g in enumerate(clipped_px_grads) if g!= None]
                avg_op = [grads.assign(grads/batched_per_lot) for grads in accum_grads]
            else:
                zero_op = None
                accum_op = None
                avg_op = None
                accum_grads = clipped_px_grads
            
            # free bugs
            
            clipped_px_gradients = []
            for g_, var in zip(accum_grads, opt_vars):
                if len(var.shape) == 1 and len(g_.shape) == 4:
                    g_ = tf.reduce_mean(g_, axis=[1,2])
                clipped_px_gradients.append(g_)
            
            # Add noise
            ratio = px_clipping_norm/lot_size
            def noise():
                grads = []
                for px_grad in clipped_px_gradients:
                    grads.append(dp_utils.AddGaussianNoise(tf.reduce_mean(px_grad, 0), dp_sigma * ratio))
                return grads
            def no_noise():
                grads = []
                for px_grad in clipped_px_gradients:
                    grads.append(tf.reduce_mean(px_grad, 0))
                return grads
            with tf.control_dependencies([privacy_accum_op]):
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


    def tf_load_pretrained(self, sess):
        print("pre trained layers loaded.")
        sess.run(self.pre_trained.pretrained())


    def tf_load(self, sess, scope=None, name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        if scope == None:
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list)
        path = FLAGS.CNN_PATH
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))


    def tf_save(self, sess, scope=None, name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        if scope == None:
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        else:
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list)
        path = FLAGS.CNN_PATH
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(sess, path +'/'+name)
        print("Save model to {}".format(path +'/'+name))