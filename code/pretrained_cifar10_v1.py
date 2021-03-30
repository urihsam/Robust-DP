
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensornets as nets
from dependency import *
import numpy as np
import os
from tensornets.preprocess import keras_resnet_preprocess
from utils.data_utils_cifar10 import dataset


data = dataset()
# In[10]:


batch_size = 32
num_epoch = 400
reg_scale = 1e-5
lr = 1e-4
beta = 100.0


# In[11]:


tf.reset_default_graph()
g = tf.get_default_graph()
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    label_holder = tf.placeholder(tf.float32, [batch_size, 10])
    is_training = tf.placeholder(tf.bool, ())
    
    # cnn model
    #x_ = tf.image.resize(x_holder, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    #x_ = tf.image.pad_to_bounding_box(x_holder, 16, 16, 64, 64)
    x_ = x_holder# + tf.random.normal(tf.shape(x_holder), mean=0.0, stddev=0.01)
    #
    label = label_holder
    pre_trained = nets.VGG16(x_, is_training=is_training, stem=True)
    #import pdb; pdb.set_trace()
    #pre_trained_cnn = tf.reshape(pre_trained.outputs()[-9], [pre_trained.get_shape().as_list()[0], -1])
    pre_trained_cnn = tf.reshape(pre_trained, [pre_trained.get_shape().as_list()[0], -1])

    
    drop_rate = 0.3
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    with tf.variable_scope('top_layers') as opt_scope_1:
        w_1 = tf.get_variable(initializer=initializer, shape=(pre_trained_cnn.get_shape().as_list()[-1], 4096), name="w_1")
        b_1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(4096), name="b_1")
        w_2 = tf.get_variable(initializer=initializer, shape=(4096, 4096), name="w_2")
        b_2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(4096), name="b_2")
        w_3 = tf.get_variable(initializer=initializer, shape=(4096, 10), name="w_3")
        b_3 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(10), name="b_3")
        
        net = pre_trained_cnn
        #net = ne.layer_norm(self.noised_pretrain, self.is_training)
        net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_1), b_1))
        net = ne.drop_out(net, drop_rate, is_training)
        net = ne.layer_norm(net, is_training)

        net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
        net = ne.drop_out(net, drop_rate, is_training)
        net = ne.layer_norm(net, is_training)

        logits = tf.add(tf.matmul(net, w_3), b_3)
        #self.cnn_logits = tf.add(tf.matmul(net, w_2), b_2)
        prediction = tf.nn.softmax(logits)
        
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
        # loss function applied to the last layer
        # train on the loss (Adam Optimizer is used)
        loss = beta * tf.losses.softmax_cross_entropy(label, logits)
        #
        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #
        if reg_scale > 0.0:
            regularize = tf.contrib.layers.l1_l2_regularizer(reg_scale, reg_scale)
            #print tf.GraphKeys.TRAINABLE_VARIABLES
            reg_term = sum([regularize(param) for param in opt_vars])
            loss += reg_term

        global_step = tf.Variable(0, name="OPT_GLOBAL_STEP", trainable=False)
        # decay
        lr = tf.train.exponential_decay(lr, global_step, 5000, 0.98)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step, var_list=opt_vars)

        


# In[12]:


pre_trained.outputs()


# In[13]:


tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#import pdb; pdb.set_trace()


# In[14]:


tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


# In[ ]:





# In[ ]:

max_test_acc = 0.0


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=g) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Loading the parameters
    sess.run(pre_trained.pretrained())
    
    for ep in range(num_epoch):
        data.shuffle_train()
        for b_idx in range(data.train_size//batch_size):
            feed_dict = {
                x_holder: data.x_train[b_idx*batch_size:(b_idx+1)*batch_size],
                label_holder: data.y_train[b_idx*batch_size:(b_idx+1)*batch_size],
                is_training: True
            }
            sess.run(fetches=train_op, feed_dict=feed_dict)
            if b_idx % 200 == 199:
                lr_, l, acc = sess.run(fetches=[lr, loss, accuracy], feed_dict=feed_dict)
                print("Epoch %d: learning rate %f\ttrain loss %.4f\ttrain accuracy %.4f"%(ep, lr_, l, acc))
        accs = []; ls =[]
        for b_idx in range(data.test_size//batch_size):
            feed_dict = {
                x_holder: data.x_test[b_idx*batch_size:(b_idx+1)*batch_size],
                label_holder: data.y_test[b_idx*batch_size:(b_idx+1)*batch_size],
                is_training: False
            }
            l, acc = sess.run(fetches=[loss, accuracy], feed_dict=feed_dict)
            ls.append(l)
            accs.append(acc)
        print()
        print("Epoch %d: test loss %.4f\ttest accuracy %.4f"%(ep, sum(ls)/len(ls), sum(accs)/len(accs)))
        print("Test Accuracy after Optimization: {}".format(max_test_acc))
        if sum(accs)/len(accs) > max_test_acc:
            max_test_acc = sum(accs)/len(accs)

            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(var_list=var_list)
            path = "../models/CIFAR10/PRE/CNN"
            name = "model_v1.ckpt"
            if not os.path.exists(path):
                os.mkdir(path)
            saver.save(sess, path +'/'+name)
            print("Save model to {}".format(path +'/'+name))
            
        print()


    print("Test Accuracy after Optimization: {}".format(max_test_acc))
    
    



