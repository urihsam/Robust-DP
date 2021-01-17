# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Differentially private optimizers.
"""
from __future__ import division

import numpy as np
import tensorflow as tf
from dependency import *
#from gradients import gradients_impl, jacobian
from utils.vectorized_map import vectorized_map
from utils.map_fn import map_fn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients


class DPGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """Differentially private gradient descent optimizer.
  """

  def __init__(self, learning_rate, eps_delta, sanitizer,
               sigma=None, use_locking=False, name="DPGradientDescent",
               batches_per_lot=1):
    """Construct a differentially private gradient descent optimizer.

    The optimizer uses fixed privacy budget for each batch of training.

    Args:
      learning_rate: for GradientDescentOptimizer.
      eps_delta: EpsDelta pair for each epoch.
      sanitizer: for sanitizing the graident.
      sigma: noise sigma. If None, use eps_delta pair to compute sigma;
        otherwise use supplied sigma directly.
      use_locking: use locking.
      name: name for the object.
      batches_per_lot: Number of batches in a lot.
    """

    super(DPGradientDescentOptimizer, self).__init__(learning_rate,
                                                     use_locking, name)

    # Also, if needed, define the gradient accumulators
    self._batches_per_lot = batches_per_lot
    self._grad_accum_dict = {}
    if batches_per_lot > 1:
      self._batch_count = tf.Variable(1, dtype=tf.int32, trainable=False,
                                      name="batch_count")
      var_list = tf.trainable_variables()
      with tf.variable_scope("grad_acc_for"):
        for var in var_list:
          v_grad_accum = tf.Variable(tf.zeros_like(var),
                                     trainable=False,
                                     name=utils.GetTensorOpName(var))
          self._grad_accum_dict[var.name] = v_grad_accum

    self._eps_delta = eps_delta
    self._sanitizer = sanitizer
    self._sigma = sigma

  def compute_sanitized_gradients(self, loss, var_list=None,
                                  add_noise=True):
    """Compute the sanitized gradients.

    Args:
      loss: the loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """

    self._assert_valid_dtypes([loss])

    xs = [tf.convert_to_tensor(x) for x in var_list]
    px_grads = per_example_gradients.PerExampleGradients(loss, xs)
    sanitized_grads = []
    if isinstance(self._sigma, list):
      idx = 0
      for px_grad, v in zip(px_grads, var_list):
        tensor_name = utils.GetTensorOpName(v)
        sanitized_grad = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=self._sigma[idx],
            tensor_name=tensor_name, add_noise=add_noise,
            num_examples=self._batches_per_lot * tf.slice(
                tf.shape(px_grad), [0], [1]))
        sanitized_grads.append(sanitized_grad)
        idx += 1
    else:
      for px_grad, v in zip(px_grads, var_list):
        tensor_name = utils.GetTensorOpName(v)
        sanitized_grad = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=self._sigma,
            tensor_name=tensor_name, add_noise=add_noise,
            num_examples=self._batches_per_lot * tf.slice(
                tf.shape(px_grad), [0], [1]))
        sanitized_grads.append(sanitized_grad)
    return sanitized_grads


  def compute_sanitized_gradients_from_input_perturbation(self, loss, ex, input_sigma, var_list,
                                                          add_noise=True):
    """Compute the sanitized gradients.

    Args:
      loss: the loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
      eps_delta: [epsilon, delta]
      input_sigma: input_sigma
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """
    self._assert_valid_dtypes([loss])
    #import pdb; pdb.set_trace()
    xs = [tf.convert_to_tensor(x) for x in var_list]
    #import pdb; pdb.set_trace()
    # Each element in px_grads is the px_grad for a param matrix, having the shape of [batch_size, shape of param matrix]
    px_grads = per_example_gradients.PerExampleGradients(loss, xs)
    # calculate sigma, sigma has the shape of [batch_size]
    
    px_pp_grads = []
    unmasked_sigmas = []
    sigmas = []
    sanitized_grads = []

    num = 0
    for px_grad, v in zip(px_grads, var_list):
      num += 1
      if num > FLAGS.ACCOUNT_NUM:
        break
      #px_grad = utils.BatchClipByL2norm(px_grad, FLAGS.DP_GRAD_CLIPPING_L2NORM/ FLAGS.BATCH_SIZE)
      px_grad_vec = tf.reshape(px_grad, [tf.shape(px_grad)[0], -1]) # [batch_size, vec_param]
      #import pdb; pdb.set_trace()
      # method 1
      px_pp_grad = batch_jacobian(px_grad_vec, ex, use_pfor=False, parallel_iterations=px_grad_vec.get_shape().as_list()[0]*px_grad_vec.get_shape().as_list()[1]) # [b, vec_param, ex_shape]
      #px_pp_grad = tf.stop_gradient(px_pp_grad)
      px_pp_grad = tf.identity(tf.reshape(px_pp_grad, [px_pp_grad.get_shape().as_list()[0], px_pp_grad.get_shape().as_list()[1],-1])) #[b, vec_param, ex_size]
      px_pp_grads.append(px_pp_grad)
    
      #px_pp_cov = tf.identity(tf.matmul(px_pp_grad, px_pp_grad, transpose_b=True, name="mat_{}".format(num))) # [b, vec_param, vec_param]
      #px_pp_L = tf.identity(tf.linalg.cholesky(px_pp_cov, name="cho_{}".format(num))) # [b, vec_param, vec_param]
      #px_pp_L_inv = tf.identity(tf.linalg.inv(px_pp_L, name="inv_{}".format(num))) # [b, vec_param, vec_param]
      #px_pp_sen_norm = tf.identity(tf.norm(px_pp_L_inv, ord="fro", axis=[1, 2], name="fro_{}".format(num))) # [b] 
      #sigma = tf.identity(input_sigma / px_pp_sen_norm) #[batch_size]
      
      px_pp_A_norm = tf.norm(px_pp_grad, ord="fro", axis=[1, 2], name="fro_{}".format(num))
      px_pp_I_norm = tf.norm(tf.eye(px_pp_grad.get_shape().as_list()[1]), ord="fro", axis=[0, 1], name="fro_{}".format(num))
      sigma = input_sigma * px_pp_A_norm/px_pp_I_norm
      
      
      ##
      '''
      px_scale = tf.reduce_sum(tf.square(px_pp_grad), 2) # [batch_size, vec_param]  
      
      # heterogeneous: each param has different scale
      scale = tf.reduce_mean(px_scale, 1) # [batch_size]
      # minimum
      #scale = tf.reduce_min(px_scale, 1) # [batch_size]
      
      sigma = tf.sqrt(scale) * input_sigma #[batch_size]
      '''
      ##
      unmasked_sigmas.append(sigma)

      mask = tf.cast(tf.greater_equal(sigma, tf.constant(FLAGS.INPUT_DP_SIGMA_THRESHOLD)), tf.float32)
      sigma = tf.identity(sigma * mask)
      sigmas.append(sigma) 
      #
      tensor_name = utils.GetTensorOpName(v)
      px_grad = tf.identity(px_grad)
      sanitized_grad = self._sanitizer.sanitize(
          px_grad, self._eps_delta, sigma=sigma,
          tensor_name=tensor_name, add_noise=add_noise,
          num_examples=tf.slice(tf.shape(px_grad), [0], [1]),
          no_clipping=False)
      sanitized_grads.append(sanitized_grad)
    
    while num <= len(var_list):
      sigmas.append(tf.zeros([ex.get_shape().as_list()[0]]))
      num += 1
    return sanitized_grads, sigmas, unmasked_sigmas, (px_pp_A_norm, px_pp_I_norm)#(px_pp_grads, px_pp_cov, px_pp_L, px_pp_L_inv, px_pp_sen_norm)
  

  def minimize(self, loss, global_step=None, var_list=None,
                name=None):
    """Minimize using sanitized gradients.

    This gets a var_list which is the list of trainable variables.
    For each var in var_list, we defined a grad_accumulator variable
    during init. When batches_per_lot > 1, we accumulate the gradient
    update in those. At the end of each lot, we apply the update back to
    the variable. This has the effect that for each lot we compute
    gradients at the point at the beginning of the lot, and then apply one
    update at the end of the lot. In other words, semantically, we are doing
    SGD with one lot being the equivalent of one usual batch of size
    batch_size * batches_per_lot.
    This allows us to simulate larger batches than our memory size would permit.

    The lr and the num_steps are in the lot world.

    Args:
      loss: the loss tensor.
      global_step: the optional global step.
      var_list: the optional variables.
      name: the optional name.
    Returns:
      the operation that runs one step of DP gradient descent.
    """

    # First validate the var_list

    if var_list is None:
      var_list = tf.trainable_variables()
    for var in var_list:
      if not isinstance(var, tf.Variable):
        raise TypeError("Argument is not a variable.Variable: %s" % var)

    # compute the sigma used for gradient perturbation

    # Modification: apply gradient once every batches_per_lot many steps.
    # This may lead to smaller error
    if self._batches_per_lot == 1:
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list)

      grads_and_vars = list(zip(sanitized_grads, var_list))
      # conv bias
      #import pdb; pdb.set_trace()
      grads = []
      for grad, var in grads_and_vars:
        if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
          grad = tf.reduce_mean(grad, axis=[0,1])
        grads.append(grad)
      grads_and_vars = list(zip(grads, var_list))

      self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])

      apply_grads = self.apply_gradients(grads_and_vars,
                                         global_step=global_step, name=name)
      return apply_grads

    # Condition for deciding whether to accumulate the gradient
    # or actually apply it.
    # we use a private self_batch_count to keep track of number of batches.
    # global step will count number of lots processed.

    update_cond = tf.equal(tf.constant(0),
                           tf.mod(self._batch_count,
                                  tf.constant(self._batches_per_lot)))

    # Things to do for batches other than last of the lot.
    # Add non-noisy clipped grads to shadow variables.

    def non_last_in_lot_op(loss, var_list):
      """Ops to do for a typical batch.

      For a batch that is not the last one in the lot, we simply compute the
      sanitized gradients and apply them to the grad_acc variables.

      Args:
        loss: loss function tensor
        var_list: list of variables
      Returns:
        A tensorflow op to do the updates to the gradient accumulators
      """
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=False)
      
      grads_and_vars = list(zip(sanitized_grads, var_list))
      # conv bias
      grads = []
      for grad, var in grads_and_vars:
        if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
          grad = tf.reduce_mean(grad, axis=[0,1])
        grads.append(grad)
      sanitized_grads = grads

      update_ops_list = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        update_ops_list.append(grad_acc_v.assign_add(grad))
      update_ops_list.append(self._batch_count.assign_add(1))
      return tf.group(*update_ops_list)

    # Things to do for last batch of a lot.
    # Add noisy clipped grads to accumulator.
    # Apply accumulated grads to vars.

    def last_in_lot_op(loss, var_list, global_step):
      """Ops to do for last batch in a lot.

      For the last batch in the lot, we first add the sanitized gradients to
      the gradient acc variables, and then apply these
      values over to the original variables (via an apply gradient)

      Args:
        loss: loss function tensor
        var_list: list of variables
        global_step: optional global step to be passed to apply_gradients
      Returns:
        A tensorflow op to push updates from shadow vars to real vars.
      """

      # We add noise in the last lot. This is why we need this code snippet
      # that looks almost identical to the non_last_op case here.
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=True)
      
      grads_and_vars = list(zip(sanitized_grads, var_list))
      # conv bias
      grads = []
      for grad, var in grads_and_vars:
        if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
          grad = tf.reduce_mean(grad, axis=[0,1])
        grads.append(grad)
      sanitized_grads = grads

      normalized_grads = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        # To handle the lr difference per lot vs per batch, we divide the
        # update by number of batches per lot.
        normalized_grad = tf.div(grad_acc_v.assign_add(grad),
                                 tf.to_float(self._batches_per_lot))

        normalized_grads.append(normalized_grad)

      with tf.control_dependencies(normalized_grads):
        grads_and_vars = list(zip(normalized_grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars if g is not None])
        '''
        grads = []
        for grad, var in grads_and_vars:
          if "b_conv_" in var.name:
            grad = tf.reduce_mean(grad, axis=[0,1])
          grads.append(grad)
        grads_and_vars = list(zip(grads, var_list))
        '''
        apply_san_grads = self.apply_gradients(grads_and_vars,
                                               global_step=global_step,
                                               name="apply_grads")

      # Now reset the accumulators to zero
      resets_list = []
      with tf.control_dependencies([apply_san_grads]):
        for _, acc in self._grad_accum_dict.items():
          reset = tf.assign(acc, tf.zeros_like(acc))
          resets_list.append(reset)
      resets_list.append(self._batch_count.assign_add(1))

      last_step_update = tf.group(*([apply_san_grads] + resets_list))
      return last_step_update
    # pylint: disable=g-long-lambda
    update_op = tf.cond(update_cond,
                        lambda: last_in_lot_op(
                            loss, var_list,
                            global_step),
                        lambda: non_last_in_lot_op(
                            loss, var_list))
    return tf.group(update_op)

