
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensornets as nets
from dependency import *
import numpy as np
import os
import nn.mnist_classifier as mnist_cnn
from tensornets.preprocess import keras_resnet_preprocess
from utils.data_utils_mnist_raw import dataset


data = dataset("../../mnist", normalize=True, biased=False)
# In[10]:


batch_size = 32
num_epoch = 200
reg_scale = 1e-5
lr = 1e-4
beta = 100.0


# In[11]:


tf.reset_default_graph()
g = tf.get_default_graph()
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
    label_holder = tf.placeholder(tf.float32, [batch_size, 10])
    is_training = tf.placeholder(tf.bool, ())

    with tf.variable_scope('mnistcnn'):
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
        cnn_logits, cnn_prediction = cnn.prediction(x_holder)
        accuracy = cnn.accuracy(cnn_prediction, label_holder)

        
        # loss function applied to the last layer
        # train on the loss (Adam Optimizer is used)
        loss = beta * tf.losses.softmax_cross_entropy(label_holder, cnn_logits)
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
    
    for ep in range(num_epoch):
        data.train_shuffle()
        for b_idx in range(data.train_size//batch_size):
            x_train, y_train = data.next_train_batch(batch_size)
            feed_dict = {
                x_holder: x_train,
                label_holder: y_train,
                is_training: True
            }
            sess.run(fetches=train_op, feed_dict=feed_dict)
            if b_idx % 200 == 199:
                lr_, l, acc = sess.run(fetches=[lr, loss, accuracy], feed_dict=feed_dict)
                print("Epoch %d: learning rate %f\ttrain loss %.4f\ttrain accuracy %.4f"%(ep, lr_, l, acc))
        accs = []; ls =[]
        for b_idx in range(data.test_size//batch_size):
            x_test, y_test = data.next_test_batch(batch_size)
            feed_dict = {
                x_holder: x_test,
                label_holder: y_test,
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
            path = "../models/MNIST/PRE/CNN"
            name = "model_v1.ckpt"
            if not os.path.exists(path):
                os.mkdir(path)
            saver.save(sess, path +'/'+name)
            print("Save model to {}".format(path +'/'+name))
            
        print()


    print("Test Accuracy after Optimization: {}".format(max_test_acc))
    
    



