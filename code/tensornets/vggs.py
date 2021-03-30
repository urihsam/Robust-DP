"""Collection of VGG variants

The reference paper:

 - Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015
 - Karen Simonyan, Andrew Zisserman
 - https://arxiv.org/abs/1409.1556

The reference implementation:

1. Keras
 - https://github.com/keras-team/keras/blob/master/keras/applications/vgg{16,19}.py
2. Caffe VGG
 - http://www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .layers import conv2d
from .layers import dropout
from .layers import flatten
from .layers import fc
from .layers import max_pool2d
from .layers import convrelu as conv

from .ops import *
from .utils import set_args
from .utils import var_scope


if tf_later_than('2'):
    from tensorflow.contrib_layers import batch_norm
    from tensorflow.contrib_layers import layer_norm
else:
    from tensorflow.contrib.layers import batch_norm
    from tensorflow.contrib.layers import layer_norm


def __args__(is_training):
    return [([conv2d], {'padding': 'SAME', 'activation_fn': None,
                        'scope': 'conv'}),
            ([dropout], {'is_training': is_training}),
            ([flatten], {'scope': 'flatten'}),
            ([fc], {'activation_fn': None, 'scope': 'fc'}),
            ([max_pool2d], {'scope': 'pool'})]

@var_scope('stack')
def _stack_va(x, filters, blocks, is_training, scope=None):
    for i in range(1, blocks+1):
        x = conv(x, filters, 3, scope=str(i))
    x = max_pool2d(x, 2, stride=2)
    return x


def vgg_vanillia(x, blocks, is_training, classes, stem, early_stem=False, late_stem=False, use_logits=False, scope=None, reuse=None):
    x = _stack_va(x, 64, blocks[0], is_training=is_training, scope='conv1')
    x = _stack_va(x, 128, blocks[1], is_training=is_training, scope='conv2')
    x = _stack_va(x, 256, blocks[2], is_training=is_training, scope='conv3')
    x = _stack_va(x, 512, blocks[3], is_training=is_training, scope='conv4')
    x = _stack_va(x, 512, blocks[4], is_training=is_training, scope='conv5')
    if stem: return x
    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop6')
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop7')
    if late_stem: return x
    logits = fc(x, classes, scope='logits')
    if use_logits:
        return logits
    probs = softmax(logits, name='probs')
    return probs


@var_scope('stack')
def _stack(x, filters, blocks, is_training, scope=None):
    x = conv(x, filters, 3, scope=str(1))
    x = tf.nn.leaky_relu(x)
    x = dropout(x, keep_prob=0.3, is_training=is_training, scope='{}_block_1'.format(scope))
    #if tf.random.uniform(shape=[], minval=0, maxval=1)
    #res = x
    for i in range(2, blocks+1):
        res = x
        x = conv(x, filters, 3, scope=str(i))
        x = tf.nn.leaky_relu(x)
        x = dropout(x, keep_prob=0.3, is_training=is_training, scope='{}_block_{}'.format(scope, i)) 
        x += res
    #x += res
    #x = res
    #x = dropout(x, keep_prob=0.5, is_training=is_training, scope='{}_block'.format(scope))
    x = max_pool2d(x, 2, stride=2)
    #x = layer_norm(x)
    return x


def vgg(x, blocks, is_training, classes, stem, early_stem=False, late_stem=False, use_logits=False, scope=None, reuse=None):
    x = _stack(x, 64, blocks[0], is_training=is_training, scope='conv1')
    x = _stack(x, 128, blocks[1], is_training=is_training, scope='conv2')
    x = _stack(x, 256, blocks[2], is_training=is_training, scope='conv3')
    x = _stack(x, 512, blocks[3], is_training=is_training, scope='conv4')
    x = _stack(x, 512, blocks[4], is_training=is_training, scope='conv5')
    if stem: return x
    x = flatten(x)
    x = fc(x, 4096, scope='fc6')
    x = relu(x, name='relu6')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop6')
    x = fc(x, 4096, scope='fc7')
    x = relu(x, name='relu7')
    x = dropout(x, keep_prob=0.5, is_training=is_training, scope='drop7')
    if late_stem: return x
    logits = fc(x, classes, scope='logits')
    if use_logits:
        return logits
    probs = softmax(logits, name='probs')
    return probs


@var_scope('vgg16')
@set_args(__args__)
def vgg16(x, is_training=False, classes=1000,
          stem=False, early_stem=False, late_stem=False, use_logits=False, vanillia=False,  scope=None, reuse=None):
    if vanillia:
        return vgg_vanillia(x, [2, 2, 3, 3, 3], is_training, classes, stem, early_stem, late_stem, scope, reuse)
    return vgg(x, [2, 2, 3, 3, 3], is_training, classes, stem, early_stem, late_stem, scope, reuse)


@var_scope('vgg19')
@set_args(__args__)
def vgg19(x, is_training=False, classes=1000,
          stem=False, early_stem=False, late_stem=False, use_logits=False, vanillia=False, scope=None, reuse=None):
    if vanillia:
        return vgg_vanillia(x, [2, 2, 4, 4, 4], is_training, classes, stem, early_stem, late_stem, scope, reuse)
    return vgg(x, [2, 2, 4, 4, 4], is_training, classes, stem, early_stem, late_stem, scope, reuse)


# Simple alias.
VGG16 = vgg16
VGG19 = vgg19
