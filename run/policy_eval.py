"""
@author: dennybritz

Modified for the ComposeNet project by Saurabh Kumar.
"""

import sys
import os
import itertools
import numpy as np
import tensorflow as tf
import time
import logging

from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op


def log_results(eval_logger, iteration, rewards, steps):
  """Function that logs the reward statistics obtained by the agent.

  Args:
    logfile: File to log reward statistics.
    iteration: The current iteration.
    rewards: Array of rewards obtained in the current iteration.
  """
  eval_logger.info('Iteration : {}, mean reward = {}, mean step = {}, \
    all rewards = {}, all steps = {}'.format(iteration, np.mean(rewards),
    np.mean(steps), ','.join([str(r) for r in rewards]),
    ','.join([str(s) for s in steps])))


class PolicyEval(object):
  """
  Helps evaluating a policy by running a fixed number of episodes in an environment,
  and logging summary statistics to a text file.
  Args:
    env: environment to run in
    policy_net: A policy estimator
  """
  def __init__(self, env, policy_net, saver=None, n_eval=50, logfile=None, checkpoint_path=None):

    self.env = env
    self.global_policy_net = policy_net
    self.saver = saver
    self.n_eval = n_eval
    self.checkpoint_path = checkpoint_path
    self.logger = logging.getLogger('eval runs')
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    hdlr.setFormatter(formatter)
    self.logger.addHandler(hdlr)
    self.logger.setLevel(logging.INFO)

    # Local policy net
    with tf.variable_scope("policy_eval"):
      self.policy_net = PolicyEstimator(policy_net.num_outputs, state_dims=self.env.get_state_size())

    # Op to copy params from global policy/value net parameters
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.policy_net.states: [state] }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def eval(self, sess, n_eval):
    with sess.as_default(), sess.graph.as_default():
      # Copy params to local model
      global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

      eval_rewards = []
      episode_lengths = []

      for i in xrange(n_eval):
        # Run an episode
        done = False
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        while not done:
          action_probs = self._policy_net_predict(state, sess)
          action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
          next_state, reward, done = self.env.step(action)
          total_reward += reward
          episode_length += 1
          state = next_state
        eval_rewards.append(total_reward)
        episode_lengths.append(episode_length)

      log_results(self.logger, global_step, eval_rewards, episode_lengths)

      # if self.saver is not None:
        # tf.add_to_collection('policy_train_op', self.policy_net.train_op)
        # tf.add_to_collection('action_probs', self.policy_net.probs)
        # tf.add_to_collection('state', self.policy_net.states)

        # self.saver.save(sess, self.checkpoint_path, global_step=global_step)
      return global_step, eval_rewards, episode_lengths

  def continuous_eval(self, eval_every, sess, coord):
    """
    Continuously evaluates the policy every [eval_every] seconds.
    """
    c = 0
    try:
      while not coord.should_stop():
        global_step, rewards, episode_lengths = self.eval(sess, self.n_eval)
        # if np.percentile(rewards, 10) > 0.9:
          # c+= 1
        # else:
          # c = 0
        # if c == 5:
          # sys.stderr.write(
            # "Thread {} converged at {} steps!\n".format(
            # self.task_id, global_step))
          # coord.request_stop()
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      return
