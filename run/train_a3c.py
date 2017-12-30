"""
@author: himanshusahni
Modified from a3c code by dennybritz

A3C baseline for comparison
"""

import unittest
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from datetime import datetime

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

sys.path.insert(0, '../environments')
sys.path.insert(0, '../rl_agents/utils')
sys.path.insert(0, '../rl_agents/policy_gradient/a3c')

from estimators import ValueEstimator, PolicyEstimator
from policy_eval import PolicyEval
from worker import Worker
from utils import make_env

tf.flags.DEFINE_string("model_dir", "experiment_logs/a3c/", "Directory to save checkpoints to.")
tf.flags.DEFINE_string("env", "objects_env", "Name of environment.")
tf.flags.DEFINE_string("task", "collect_0_avoid_1", "Task. Sequence of 'collect', 'avoid','or', or 'then' followed by object id.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update.")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 10, "Evaluate the policy every N seconds.")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 5, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

# determine number of actions from environment
env_ = make_env(FLAGS.env, FLAGS.task)
VALID_ACTIONS = list(range(env_.get_num_actions()))

# Set the number of workers
NUM_WORKERS = FLAGS.parallelism

# directories for logs and model files
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", FLAGS.task)
LOG_DIR = os.path.join(MODEL_DIR, "logs", FLAGS.task)


# Optionally empty model directory
if FLAGS.reset:
 shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

with tf.device("/cpu:0"):
  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Global policy and value nets
  with tf.variable_scope("global") as vs:
    policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS), state_dims=env_.get_state_size())
    value_net = ValueEstimator(reuse=True, state_dims=env_.get_state_size())

  # Global step iterator
  global_counter = itertools.count()

  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    worker = Worker(
      name="worker_{}".format(worker_id),
      env = make_env(FLAGS.env, FLAGS.task),
      policy_net=policy_net,
      value_net=value_net,
      global_counter=global_counter,
      discount_factor = 0.99,
      max_global_steps=FLAGS.max_global_steps)
    workers.append(worker)

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=2)

  logfile = os.path.join(
    LOG_DIR,
    '{:%Y-%m-%d_%H:%M:%S}.log'.format(datetime.now()))

  # Used to occasionally evaluate the policy and save
  # statistics and checkpoint model.
  pe = PolicyEval(
    env = make_env(FLAGS.env, FLAGS.task),
    policy_net=policy_net,
    saver=saver,
    logfile=logfile,
    checkpoint_path=CHECKPOINT_DIR)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  '''
  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
  if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
  '''

  # Start worker threads
  worker_threads = []
  for worker in workers:
    worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  # Start a thread for policy eval task
  eval_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
  eval_thread.start()

  # Wait for all workers to finish
  coord.join(worker_threads)
