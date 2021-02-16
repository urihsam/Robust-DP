import keras
import numpy as np
import tensorflow as tf
import utils.net_element as ne
import tensornets as nets
from tensorflow.python.platform import flags
flags = tf.app.flags 
FLAGS = flags.FLAGS

slim = tf.contrib.slim