
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensornets as nets
import numpy as np
import os
from tensornets.preprocess import keras_resnet_preprocess
from utils.data_utils_cifar10 import dataset


data = dataset()
# In[10]:


batch_size = 16
num_epoch = 80
reg_scale = 0.0
lr = 1e-5
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
    x_ = tf.image.resize(x_holder, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    pre_trained = nets.VGG19(x_, is_training=is_training, classes=100, use_logits=True)
    pre_trained_cnn = tf.reshape(pre_trained, [pre_trained.get_shape().as_list()[0], -1])
    

    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    with tf.variable_scope('top_layers') as opt_scope_1:
        w_1 = tf.get_variable(initializer=initializer, shape=(pre_trained_cnn.get_shape().as_list()[-1], 32), name="w_1")
        b_1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(32), name="b_1")
        w_2 = tf.get_variable(initializer=initializer, shape=(32, 10), name="w_2")
        b_2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=(10), name="b_2")
        
        net = pre_trained_cnn
        #net = ne.layer_norm(self.noised_pretrain, self.is_training)
        net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_1), b_1))
        #net = ne.layer_norm(net, self.is_training)

        logits = tf.add(tf.matmul(net, w_2), b_2)
        #self.cnn_logits = tf.add(tf.matmul(net, w_2), b_2)
        prediction = tf.nn.softmax(logits)
        
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
        # loss function applied to the last layer
        # train on the loss (Adam Optimizer is used)
        loss = beta * tf.losses.softmax_cross_entropy(label_holder, logits)
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
        lr = tf.train.exponential_decay(lr, global_step, 10000, 0.99)
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
                x_holder: keras_resnet_preprocess(data.x_train[b_idx*batch_size:(b_idx+1)*batch_size]),
                label_holder: data.y_train[b_idx*batch_size:(b_idx+1)*batch_size],
                is_training: True
            }
            sess.run(fetches=train_op, feed_dict=feed_dict)
            if b_idx % 400 == 399:
                lr_, l, acc = sess.run(fetches=[lr, loss, accuracy], feed_dict=feed_dict)
                print("Epoch %d: learning rate %f\ttrain loss %.4f\ttrain accuracy %.4f"%(ep, lr_, l, acc))
        accs = []; ls =[]
        for b_idx in range(data.test_size//batch_size):
            feed_dict = {
                x_holder: keras_resnet_preprocess(data.x_test[b_idx*batch_size:(b_idx+1)*batch_size]),
                label_holder: data.y_test[b_idx*batch_size:(b_idx+1)*batch_size],
                is_training: False
            }
            l, acc = sess.run(fetches=[loss, accuracy], feed_dict=feed_dict)
            ls.append(l)
            accs.append(acc)
        print()
        print("Epoch %d: test loss %.4f\ttest accuracy %.4f"%(ep, sum(ls)/len(ls), sum(accs)/len(accs)))
        print()
    
    


# In[ ]:


    '''
    ex_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "top_layers")
    var_list = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var not in ex_var_list:
            var_list.append(var)
    saver = tf.train.Saver(var_list=var_list)
    path = "../models/CIFAR10/PRE/CNN"
    name = "model.ckpt"
    if not os.path.exists(path):
        os.mkdir(path)
    saver.save(sess, path +'/'+name)
    print("Save model to {}".format(path +'/'+name))
    '''

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=var_list)
    path = "../models/CIFAR10/PRE/CNN"
    name = "model.ckpt"
    if not os.path.exists(path):
        os.mkdir(path)
    saver.save(sess, path +'/'+name)
    print("Save model to {}".format(path +'/'+name))

