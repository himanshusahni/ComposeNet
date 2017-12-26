"""
@author: himanshusahni
Modified from a3c code by dennybritz
"""

import sys
import os
import itertools
import numpy as np
import tensorflow as tf
import time
import logging

from estimators import Skill, CompositionModule, PolicyModule, ValueModule
from utils import make_copy_params_op
from copy import copy

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)
sys.path.insert(0, '../environments')
sys.path.insert(0, '../rl_agents/utils')
sys.path.insert(0, '../rl_agents/policy_gradient/composenet')

def log_results(eval_logger, task_id, iteration, global_step, rewards, steps):
  """Function that logs the reward statistics obtained by the agent.

  Args:
    logfile: File to log reward statistics.
    iteration: The current iteration.
    rewards: Array of rewards obtained in the current iteration.
  """
  eval_logger.info('Global Step : {}, Iteration : {}, environment : {}, mean reward = {},\
    mean step = {}, all rewards = {}, all steps = {}'.format(global_step, iteration, task_id,\
    np.mean(rewards), np.mean(steps), ','.join([str(r) for r in rewards]), \
    ','.join([str(s) for s in steps])))

class PolicyEval(object):
  """
  Helps evaluating a policy by running a fixed number of episodes in an environment,
  and logging summary statistics to a text file.
  Args:
    env: environment to run in
    policy_net: A policy estimator
  """
  def __init__(self, task_id, env, policy_net, global_scopes, task_counter,
      global_counter, saver=None, n_eval=50, logfile=None,
      checkpoint_path=None):

    self.env = env
    self.task_id = task_id
    self.global_policy_net = policy_net
    self.saver = saver
    self.n_eval = n_eval
    self.counter = task_counter
    self.global_counter = global_counter
    self.checkpoint_path = checkpoint_path
    self.logger = logging.getLogger('eval runs {}'.format(task_id))
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    hdlr.setFormatter(formatter)
    self.logger.addHandler(hdlr)
    self.logger.setLevel(logging.INFO)
    self.converged = False
    self.global_scopes = global_scopes

  def create_copy_ops(self, key_func):
    '''
    hook op global/local copy
    '''
    global_variables = []
    for gs in self.global_scopes:
      global_variables += tf.contrib.slim.get_variables(
        scope=gs,
        collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    local_variables = tf.contrib.slim.get_variables(
      scope="policy_eval_{}/".format(self.task_id),
      collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    self.copy_params_op = make_copy_params_op(
      global_variables, local_variables, key_func)

  def _policy_net_predict(self, state, sess):
    # feed in the state to all embedders
    feed_dict = {}
    for i in range(len(self.policy_net.states)):
      feed_dict[self.policy_net.states[i]] = [state]
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def eval(self, sess, n_eval):
    with sess.as_default(), sess.graph.as_default():
      # Copy params to local model
      sess.run(self.copy_params_op)
      task_step = copy(self.counter)
      global_step = copy(self.global_counter)

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

      log_results(self.logger,
        self.task_id,
        next(task_step),
        next(global_step),
        eval_rewards,
        episode_lengths)

      if self.saver is not None:
        if self.task_id == 0:
          self.saver.save(sess, self.checkpoint_path, global_step=next(global_step))
    return next(task_step)-1, eval_rewards, episode_lengths


  def continuous_eval(self, eval_every, sess, coord, converged_threads):
    """
    Continuously evaluates the policy every [eval_every] seconds.
    """
    try:
      while not coord.should_stop():
        task_step, rewards, episode_lengths = self.eval(sess, self.n_eval)
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      return
