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
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients


class DPGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """Differentially private gradient descent optimizer.
  """

  def __init__(self, learning_rate, eps_delta, sanitizer,
               sigma=None, accountant_sigma=None, use_locking=False, name="DPGradientDescent",
               batches_per_lot=1, 
               is_sigma_layerwised=False, is_sigma_data_dependent=False):
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

    self._is_sigma_layerwised = is_sigma_layerwised
    self._is_sigma_data_dependent = is_sigma_data_dependent
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
    self._act_sigma = accountant_sigma

  def compute_sanitized_gradients(self, loss_list, var_list=None,
                                  add_noise=True):
    """Compute the sanitized gradients.

    Args:
      loss_list: the list of loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """
    for loss in loss_list:
      self._assert_valid_dtypes([loss])

    xs = [tf.convert_to_tensor(x) for x in var_list]
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
    
    sanitized_grads = []
    idx = 0
    for px_grad, v in zip(px_grads, var_list):
      tensor_name = utils.GetTensorOpName(v)
      if self._is_sigma_layerwised:
        sig = self._sigma[idx]
        if self._act_sigma != None:
          act_sig = self._act_sigma[idx]
      else:
        sig = self._sigma
        if self._act_sigma != None:
          act_sig = self._act_sigma
      
      def no_noise():
        act_op = self._sanitizer.sanitize(
          px_grad, self._eps_delta, sigma=act_sig,
          tensor_name=tensor_name, is_sigma_scalar=not self._is_sigma_data_dependent, add_noise=add_noise,
          num_examples=self._batches_per_lot * tf.slice(
              tf.shape(px_grad), [0], [1]))
        with tf.control_dependencies([act_op]):
          sanitized_grad = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=sig,
            tensor_name=tensor_name, is_sigma_scalar=not self._is_sigma_data_dependent, add_noise=False,
            num_examples=self._batches_per_lot * tf.slice(
                tf.shape(px_grad), [0], [1]))
        
        return sanitized_grad
      
      def noise():
        act_op = self._sanitizer.sanitize(
          px_grad, self._eps_delta, sigma=act_sig,
          tensor_name=tensor_name, is_sigma_scalar=not self._is_sigma_data_dependent, add_noise=add_noise,
          num_examples=self._batches_per_lot * tf.slice(
              tf.shape(px_grad), [0], [1]))
        with tf.control_dependencies([act_op]):
          sanitized_grad = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=sig,
            tensor_name=tensor_name, is_sigma_scalar=not self._is_sigma_data_dependent, add_noise=add_noise, no_account=True,
            num_examples=self._batches_per_lot * tf.slice(
                tf.shape(px_grad), [0], [1]))
        return sanitized_grad
      if self._act_sigma != None:
        sanitized_grad = tf.cond(tf.equal(sig, tf.constant(0.0)), no_noise, noise)
      else:
        sanitized_grad = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=sig,
            tensor_name=tensor_name, is_sigma_scalar=not self._is_sigma_data_dependent, add_noise=add_noise,
            num_examples=self._batches_per_lot * tf.slice(
                tf.shape(px_grad), [0], [1]))
      sanitized_grads.append(sanitized_grad)
      idx += 1
    return sanitized_grads
    

  def minimize(self, loss_list, global_step=None, var_list=None,
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
      loss_list: the list of loss tensor.
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

    #import pdb; pdb.set_trace()
    # compute the sigma used for gradient perturbation

    # Modification: apply gradient once every batches_per_lot many steps.
    # This may lead to smaller error
    if self._batches_per_lot == 1:
      sanitized_grads = self.compute_sanitized_gradients(
          loss_list, var_list=var_list)

      grads_and_vars = list(zip(sanitized_grads, var_list))
      # conv bias
      #import pdb; pdb.set_trace()
      grads = []
      for grad, var in grads_and_vars:
        #if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
        if len(var.shape) == 1 and len(grad.shape) == 3:
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
        #import pdb; pdb.set_trace()
        #if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
        if len(var.shape) == 1 and len(grad.shape) == 3:
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
        #import pdb; pdb.set_trace()
        #if "b_conv_" in var.name or ("b_g" in var.name and "_res"  in var.name):
        if len(var.shape) == 1 and len(grad.shape) == 3:
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
                            loss_list, var_list,
                            global_step),
                        lambda: non_last_in_lot_op(
                            loss_list, var_list))
    return tf.group(update_op)

