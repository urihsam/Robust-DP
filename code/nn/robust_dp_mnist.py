import nn.mnist_classifier as mnist_cnn
from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer_v2 as dp_optimizer
from utils.decorator import *
from dependency import *
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

        # cnn model
        with tf.variable_scope('CNN') as scope:
            self.cnn = mnist_cnn.MNISTCNN(conv_filter_sizes=[[4,4], [4,4], [4,4]],
                        conv_strides = [[2,2], [2,2], [2,2]], 
                        conv_channel_sizes=[8, 16, 32], 
                        conv_leaky_ratio=[0.2, 0.2, 0.2],
                        conv_drop_rate=[0.0, 0.2, 0.0],
                        conv_residual=True,
                        num_res_block=1,
                        res_block_size=1,
                        out_state=4*4*32,
                        out_fc_states=[10],
                        out_leaky_ratio=0.2,
                        out_norm="NONE",
                        use_norm="NONE",
                        img_channel=1)
            self.cnn_logits, self.cnn_prediction = self.cnn.prediction(self.noised_data)
            self.cnn_accuracy = self.cnn.accuracy(self.cnn_prediction, self.label)

        # recon
        with tf.variable_scope(scope, reuse=True):
            self.cnn_clean = self.cnn
            self.cnn_clean_logits, self.cnn_clean_prediction = self.cnn_clean.prediction(self.data)
            self.cnn_clean_accuracy = self.cnn_clean.accuracy(self.cnn_clean_prediction, self.label)


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
        loss = FLAGS.BETA * self.cnn.loss(self.cnn_logits, self.label, loss_type="xentropy")
        tf.summary.scalar("Total_loss", loss)
        return loss


    @lazy_method
    def loss_clean(self):
        loss = FLAGS.BETA * self.cnn_clean.loss(self.cnn_clean_logits, self.label, loss_type="xentropy")
        tf.summary.scalar("Total_loss_clean", loss)
        return loss


    @lazy_method_no_scope
    def dp_accountant(self, loss, sanitizer, dp_sigma, learning_rate, scope="DP_ACCT"):
        with tf.variable_scope(scope):
            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CNN")
            
            # dp opt
            optimizer = dp_optimizer.DPGradientDescentOptimizer(
                learning_rate,
                [None, None],
                sanitizer,
                sigma=dp_sigma,
                batches_per_lot=FLAGS.BATCHES_PER_LOT)
            
            op, sigma_used, unmasked, res = optimizer.compute_sanitized_gradients_from_input_perturbation(loss, self.noised_data, 
                                                                                input_sigma=self.input_sigma, 
                                                                                var_list=opt_vars)

            return op, sigma_used, unmasked, res

    
    def vars_size(self):
        return len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CNN"))


    @lazy_method_no_scope
    def dp_optimization(self, loss, sanitizer, dp_sigma, batched_per_lot=1, scope="DP_OPT"):
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

                opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CNN")
                
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

            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CNN")
            
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

            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CNN")

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


    def tf_load(self, sess, scope='CNN', name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = FLAGS.CNN_PATH+'/'+scope
        if not os.path.exists(path):
            print("Wrong path: {}".format(path))
        saver.restore(sess, path +'/'+name)
        print("Restore model from {}".format(path +'/'+name))


    def tf_save(self, sess, scope='CNN', name='robust_dp_cnn.ckpt'):
        #saver = tf.train.Saver(dict(self.conv_filters, **self.conv_biases, **self.decv_filters, **self.decv_biases))
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        path = FLAGS.CNN_PATH+'/'+scope
        if not os.path.exists(path):
            os.mkdir(path)
        saver.save(sess, path +'/'+name)
        print("Save model to {}".format(path +'/'+name))
