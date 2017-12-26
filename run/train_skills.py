"""
@author: himanshusahni
Modified from a3c code by dennybritz

Train base skill embedding modules.
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
sys.path.insert(0, '../rl_agents/policy_gradient/composenet')
import objects_env
from make_env import make_env

from estimators import Skill, CompositionModule, PolicyModule, ValueModule
from policy_eval_composenet import PolicyEval
from worker import Worker

tf.flags.DEFINE_string("model_dir", "experiment_logs/a4c", "Directory to save checkpoints to.")
tf.flags.DEFINE_string("env", "objects_env", "Name of environment.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update.")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 10, "Evaluate the policy every N seconds.")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 30, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

def key_func(var):
  """
  key function for sorting in the graph structures that ensures weights are
  copied and gradients are applied correctly.
  Any global/worker scopes can be ignored for sorting.
  Args:
      var: tensorflow variable
  """
  if 'worker' in var.name.split('/')[0] or \
      'global' in var.name.split('/')[0] or \
      'policy_eval' in var.name.split('/')[0]:
    return '/'.join(var.name.split('/')[1:])
  else:
    return var.name

# skills depend on environment
if FLAGS.env == 'objects_env':
  NUM_OBJECTS = objects_env.World.NUM_OBJECTS
  SKILLS = []
  for i in range(NUM_OBJECTS):
    SKILLS += ['collect_{}'.format(i), 'evade_{}'.format(i)]
NUM_TASKS = len(SKILLS)

# Depending on the game we may have a limited action space
env_ = make_env(FLAGS.env, SKILLS[0])
VALID_ACTIONS = list(range(env_.get_num_actions()))

# Set the number of workers
NUM_WORKERS = FLAGS.parallelism

# directories for logs and model files
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", FLAGS.env)
LOG_DIR = os.path.join(MODEL_DIR, "logs", FLAGS.env)

# Optionally empty model directory
if FLAGS.reset:
 shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

# create the environments to learn skills
envs = []
for worker_id in range(NUM_WORKERS):
  task = SKILLS[worker_id % NUM_TASKS]
  envs.append(make_env(FLAGS.env, task))

with tf.device("/cpu:0"):
  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)
  task_steps = [tf.Variable(0, name="task_step_{}".format(i), trainable=False) \
    for i in range(NUM_TASKS)]

  # Global policy and value nets for each task
  global_policy_nets = []
  global_value_nets = []
  for task_id in range(NUM_TASKS):
    skill_embedder = Skill(
      name_scope='global_{}'.format(task_id),
      state_dims=env_.get_state_size())
    policy_net = PolicyModule(
      name_scope='global_{}'.format(task_id),
      trainable_scopes=['global_{}/'.format(task_id), 'policy_net/'],
      num_outputs=len(VALID_ACTIONS),
      embedder=skill_embedder,
      global_final_layer=True)
    value_net = ValueModule(
      name_scope='global_{}'.format(task_id),
      trainable_scopes=['global_{}/'.format(task_id), 'value_net/'],
      embedder=skill_embedder,
      global_final_layer=True)
    # add to the global list
    global_policy_nets.append(policy_net)
    global_value_nets.append(value_net)

  # Global step iterator
  global_counter = itertools.count()
  task_counters = [itertools.count() for i in range(NUM_TASKS)]

  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    task_id = worker_id % NUM_TASKS
    # need curriculum?
    if task_id%2 == 1:
      curriculum = [5, 6, 7, 8, 9, 10]
    else:
      curriculum = [10]
    # global scopes to copy weights from
    global_scopes = ["global_{}/".format(task_id),
      "policy_net/",
      "value_net/"]
    # create the worker
    worker = Worker(
      name="worker_{}".format(worker_id),
      task_id=task_id,
      env=envs[worker_id],
      policy_net=global_policy_nets[task_id],
      value_net=global_value_nets[task_id],
      global_scopes=global_scopes,
      global_counter=global_counter,
      task_counter=task_counters[task_id],
      task_step=task_steps[task_id],
      curriculum=curriculum,
      max_global_steps=FLAGS.max_global_steps)
    # create local worker graph structure
    worker.skill_embedder = Skill(
      name_scope="worker_{}".format(worker_id),
      state_dims=env_.get_state_size())
    worker.policy_net = PolicyModule(
      name_scope="worker_{}".format(worker_id),
      trainable_scopes=['worker_{}/'.format(worker_id)],
      num_outputs=len(VALID_ACTIONS),
      embedder=worker.skill_embedder,
      global_final_layer=False)
    worker.value_net = ValueModule(
      name_scope="worker_{}".format(worker_id),
      trainable_scopes=['worker_{}/'.format(worker_id)],
      embedder=worker.skill_embedder,
      global_final_layer=False)
    # create ops to copy global network into worker network
    worker.create_copy_ops(key_func)
    # create the gradient update ops in the worker
    worker.create_train_ops(key_func)
    workers.append(worker)

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=2)

  logfile = os.path.join(
    LOG_DIR,
    '{:%Y-%m-%d_%H:%M:%S}.log'.format(datetime.now()))

  # Used to occasionally evaluate the policy and save
  # statistics and checkpoint model.
  evaluators = []
  for i, task in enumerate(SKILLS):
    # global scopes to copy weights from
    global_scopes = ["global_{}/".format(i),
      "policy_net/",]
    ev = PolicyEval(
      task_id=i,
      env=make_env(FLAGS.env, task),
      policy_net=global_policy_nets[i],
      global_scopes=global_scopes,
      task_counter=task_counters[i],
      global_counter=global_counter,
      saver=saver,
      logfile=logfile,
      checkpoint_path=CHECKPOINT_DIR+'/')
    ev.skill_embedder = Skill(
      name_scope="policy_eval_{}".format(i),
      state_dims=env_.get_state_size())
    ev.policy_net = PolicyModule(
      name_scope="policy_eval_{}".format(i),
      trainable_scopes=[],
      num_outputs=len(VALID_ACTIONS),
      embedder=ev.skill_embedder,
      global_final_layer=False)
    ev.create_copy_ops(key_func)
    evaluators.append(ev)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  # Start threads for policy eval task
  eval_threads = []
  converged_threads = itertools.count()
  for ev in range(NUM_TASKS):
    eval_thread = threading.Thread(
      target=evaluators[ev].continuous_eval,
      args=(FLAGS.eval_every, sess, coord, converged_threads))
    eval_thread.start()
    eval_threads.append(eval_thread)

  # Start worker threads
  worker_threads = []
  for worker in workers:
    worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  # Wait for all workers to finish
  coord.join(worker_threads)
  coord.join(eval_threads)
