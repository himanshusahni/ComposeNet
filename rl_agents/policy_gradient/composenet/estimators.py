"""
@author: himanshusahni
Modified from a3c code by dennybritz

Defines policy/value modules used by composenet agent
"""

import numpy as np
import tensorflow as tf
from utils import *
import sys


class Skill(object):
  """
  Base skill Embedding generartor. Given a state, will generate an embedding
  for a particular skill.
  The same network is used for generating policy and value embeddings.
  Args:
    name_scope: Prepended to scope of every variable
    reuse: If true, an existing shared network will be re-used.
    state_dims: Dimensions of the input state.
  """

  def __init__(self, name_scope='', reuse=False, state_dims=None):
    with tf.variable_scope(name_scope):
      # Placeholders for our input
      self.states = [tf.placeholder(shape=[
        None, state_dims[0], state_dims[1], 1], dtype=tf.uint8, name="X")]

      # Normalize
      X = tf.to_float(self.states[0]) / 255.0
      self.batch_size = tf.shape(self.states[0])[0]

      # the graph structure
      with tf.variable_scope("shared", reuse=reuse):
        self.embedding = self.build_shared_network(X)

  def build_shared_network(self, X):
    """
    Builds a 3-layer network conv -> conv -> fc.
    This network is shared by both the policy and value network.
    Args:
      X: Inputs
    Returns:
      Final layer activations.
    """

    NUM_CONV_1_FILTERS = 10
    NUM_CONV_2_FILTERS = 20

    # Two convolutional layers.
    w1 = weight_variable([3, 3, 1, NUM_CONV_1_FILTERS], name='w1')
    b1 = bias_variable([NUM_CONV_1_FILTERS])
    conv1 = tf.nn.relu(conv2d(X, w1) + b1)

    w2 = weight_variable(
      [3, 3, NUM_CONV_1_FILTERS, NUM_CONV_2_FILTERS], name='w2')
    b2 = bias_variable([NUM_CONV_2_FILTERS])
    conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)

    # Fully connected layer
    fc1 = tf.contrib.layers.fully_connected(
      inputs=tf.contrib.layers.flatten(conv2),
      num_outputs=256,
      scope="fc1")

    return fc1

class CompositionModule(object):
  """
  Composed embedding generator. Creates a composed embedding out of two base
  embeddings
  Args:
    name_scope: Prepended to scope of every variable.
    embedder_1: first embedding object to be composed
    embedder_2: second embedding object to be composed
  """
  def __init__(
      self, name_scope, embedder_1, embedder_2):

    # assuming batch size is same for everything
    self.batch_size = embedder_1.batch_size
    # collect all the state input placeholders
    self.states = embedder_1.states + embedder_2.states
    # concatenate the two trunks together
    self.concat_layer = tf.concat([embedder_1.embedding, embedder_2.embedding], 1)
    # fully connected layer for compose
    self.embedding= tf.contrib.layers.fully_connected(
      inputs=self.concat_layer,
      num_outputs=256,
      reuse=tf.AUTO_REUSE,
      scope=name_scope+'/fully_connected')

class PolicyModule(object):
  """
  Converts an embedding into a policy, i.e. a distribution over actions
  Args:
    name_scope: Prepended to scope of loss related variables and policy layer
      for worker threads
    trainable_scopes: list of scopes to apply gradient to
    num_outputs: number of actions
    embedder: embedding object for task
    reuse: If true, an existing shared network will be re-used.
    global_final_layer: whether the policy layer is shared
  """

  def __init__(
      self, name_scope, trainable_scopes, num_outputs, embedder,
      global_final_layer=False):
    print name_scope
    print trainable_scopes

    # assuming batch size is same for everything
    self.batch_size = embedder.batch_size
    # collect all the state input placeholders
    self.states = embedder.states

    with tf.variable_scope(name_scope):
      self.num_outputs = num_outputs
      # The TD target value
      self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
      # Integer id of which action was selected
      self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    # shared policy layer
    if global_final_layer:
      self.logits = tf.contrib.layers.fully_connected(
        embedder.embedding, num_outputs, activation_fn=None,
        reuse=tf.AUTO_REUSE, scope="policy_net/fully_connected")
      with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):
        self.probs = tf.nn.softmax(self.logits) + 1e-8
    else:
      with tf.variable_scope(name_scope):
        with tf.variable_scope("policy_net"):
          self.logits = tf.contrib.layers.fully_connected(
            embedder.embedding, num_outputs, activation_fn=None)
          self.probs = tf.nn.softmax(self.logits) + 1e-8

    # get loss/gradients
    with tf.variable_scope(name_scope):
      with tf.variable_scope("policy_net"):
        self.predictions = {
          "logits": self.logits,
          "probs": self.probs
        }

        # We add entropy to the loss to encourage exploration
        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
        self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(self.batch_size) * tf.shape(self.probs)[1] + self.actions
        self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

        self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
        self.loss = tf.reduce_sum(self.losses, name="loss")

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # only take gradients for the trainable layers
        self.grads_and_vars = [[grad, var] \
          for grad, var in self.grads_and_vars \
          if grad is not None and \
          any([cs in var.name for cs in trainable_scopes])]
        if len(self.grads_and_vars) > 0:
          self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
            global_step=tf.contrib.framework.get_global_step())

class ValueModule(object):
  """
  Converts an embedding into a value.
  Args:
    name_scope: Prepended to scope of every variable
    trainable_scopes: list of scopes to apply gradient to
    embedder: embedder for task
    reuse: If true, an existing shared network will be re-used.
    state_dims: Dimensions of the input state.
  """

  def __init__(
      self, name_scope, trainable_scopes, embedder, global_final_layer=False):

    # collect all the state input placeholders
    self.states = embedder.states

    with tf.variable_scope(name_scope):
      # The TD target value
      self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

    # now add the original value net on top
    if global_final_layer:
      self.logits = tf.contrib.layers.fully_connected(
        embedder.embedding, num_outputs=1, activation_fn=None,
        reuse=tf.AUTO_REUSE, scope="value_net/fully_connected")
      with tf.variable_scope("value_net", reuse=tf.AUTO_REUSE):
        self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")
    else:
      with tf.variable_scope(name_scope):
        with tf.variable_scope("value_net"):
          self.logits = tf.contrib.layers.fully_connected(
            embedder.embedding, num_outputs=1, activation_fn=None)
          self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

    with tf.variable_scope(name_scope):
      with tf.variable_scope("value_net"):
        self.losses = tf.squared_difference(self.logits, self.targets)
        self.loss = tf.reduce_sum(self.losses, name="loss")

        self.predictions = {
          "logits": self.logits
        }

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # only take gradients for the trainable layers
        self.grads_and_vars = [[grad, var] \
          for grad, var in self.grads_and_vars \
          if grad is not None and \
          any([cs in var.name for cs in trainable_scopes])]
        if len(self.grads_and_vars) > 0:
          self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
            global_step=tf.contrib.framework.get_global_step())
