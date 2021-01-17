
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensornets as nets
import numpy as np
import os
from tensornets.preprocess import keras_resnet_preprocess


# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# In[3]:


x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 


# In[4]:


train_size = x_train.shape[0]
test_size = x_test.shape[0]


# In[5]:


def _one_hot_encode(inputs, encoded_size):
    def get_one_hot(number):
        on_hot=[0]*encoded_size
        on_hot[int(number)]=1
        return on_hot
    return list(map(get_one_hot, inputs))


# In[6]:


# one hot
y_train = np.array(_one_hot_encode(y_train, 10))
y_test = np.array(_one_hot_encode(y_test, 10))


# In[7]:


x_test[0].shape


# In[8]:


y_test.shape


# In[9]:


import random
# shuffling
train_idx = list(range(train_size))
random.shuffle(train_idx)
x_train = x_train[train_idx]
y_train = y_train[train_idx]

test_idx = list(range(test_size))
random.shuffle(test_idx)
x_test = x_test[test_idx]
y_test = y_test[test_idx]


# In[10]:


batch_size = 128
num_epoch = 80
reg_scale = 5e-5
lr = 5e-6


# In[11]:


tf.reset_default_graph()
g = tf.get_default_graph()
with g.as_default():
    # Placeholder nodes.
    x_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    label_holder = tf.placeholder(tf.float32, [batch_size, 10])
    is_training = tf.placeholder(tf.bool, ())
    
    pre_trained = nets.VGG19(x_holder, is_training=False, early_stem=True, stem=True)
    net =  tf.identity(pre_trained)
    print(net.get_shape().as_list())
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
    
    w_1 = tf.get_variable(initializer=initializer, shape=(3, 3, net.get_shape().as_list()[-1], 128), name="w_1")
    b_1 = tf.get_variable(initializer=initializer, shape=(128), name="b_1")
    net = tf.nn.conv2d(net, w_1, strides=[1, 1, 1, 1], padding="SAME")+ b_1
    net = tf.nn.leaky_relu(net)
    net = tf.reshape(net, [batch_size, -1])
    with tf.variable_scope('top_layers') as opt_scope:
        w_2 = tf.get_variable(initializer=initializer, shape=(net.get_shape().as_list()[-1], 32), name="w_2")
        b_2 = tf.get_variable(initializer=initializer, shape=(32), name="b_2")
        w_3 = tf.get_variable(initializer=initializer, shape=(32,16), name="w_3")
        b_3 = tf.get_variable(initializer=initializer, shape=(16), name="b_3")
        w_4 = tf.get_variable(initializer=initializer, shape=(16, 10), name="w_4")
        b_4 = tf.get_variable(initializer=initializer, shape=(10), name="b_4")
        
        net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_2), b_2))
        net = tf.nn.leaky_relu(tf.add(tf.matmul(net, w_3), b_3))
        logits = tf.add(tf.matmul(net, w_4), b_4)
        probs = tf.nn.softmax(logits)
        
        # loss function applied to the last layer
        # train on the loss (Adam Optimizer is used)
        loss = tf.losses.softmax_cross_entropy(label_holder, logits)
        #
        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #
        regularize = tf.contrib.layers.l1_l2_regularizer(reg_scale, reg_scale)
        #print tf.GraphKeys.TRAINABLE_VARIABLES
        reg_term = sum([regularize(param) for param in opt_vars])
        loss += reg_term
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=opt_vars)

        # for measuring accuracy after forward passing
        correct_pred = tf.equal(tf.argmax(probs, 1), tf.argmax(label_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        


# In[12]:


pre_trained.outputs()


# In[13]:


tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "top_layers")


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
        for b_idx in range(train_size//batch_size):
            feed_dict = {
                x_holder: keras_resnet_preprocess(x_train[b_idx*batch_size:(b_idx+1)*batch_size]),
                label_holder: y_train[b_idx*batch_size:(b_idx+1)*batch_size]
            }
            sess.run(fetches=train_op, feed_dict=feed_dict)
            if b_idx % 100 == 99:
                acc = sess.run(fetches=accuracy, feed_dict=feed_dict)
                print("Epoch %d: train accuracy %.4f"%(ep, acc))
        accs = []
        for b_idx in range(test_size//batch_size):
            feed_dict = {
                x_holder: keras_resnet_preprocess(x_test[b_idx*batch_size:(b_idx+1)*batch_size]),
                label_holder: y_test[b_idx*batch_size:(b_idx+1)*batch_size]
            }
            accs.append(sess.run(fetches=accuracy, feed_dict=feed_dict))
        print()
        print("Epoch %d: test accuracy %.4f"%(ep, sum(accs)/len(accs)))
        print()
    
    


# In[ ]:


    ex_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "top_layers")
    var_list = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if var not in ex_var_list:
            var_list.append(var)
    saver = tf.train.Saver(var_list=var_list)
    path = "../models/pretrained_vgg19_cifar10"
    name = "model.ckpt"
    if not os.path.exists(path):
        os.mkdir(path)
    saver.save(sess, path +'/'+name)
    print("Save model to {}".format(path +'/'+name))

