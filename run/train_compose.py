"""
@author: himanshusahni
Modified from a3c code by dennybritz

For training compositions on top of base skills
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

tf.flags.DEFINE_string("model_dir", "experiment_logs/compose", "Directory to save checkpoints to.")
tf.flags.DEFINE_string("trained_primitives", "experiment_logs/skills/", "Directory to load saved skills from.")
tf.flags.DEFINE_string("transfer_dir", None, "Directory to load pre-trained compose layer from.")
tf.flags.DEFINE_string("env", "objects_env", "Name of environment.")
tf.flags.DEFINE_string("task", "collect_0_evade_1", "Name of environment.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update.")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 2, "Evaluate the policy every N seconds.")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 5, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

def key_func(var):
  """
  key function for sorting in the graph structures that ensures weights are
  copied and gradients are applied correctly.
  Here, it should ignore only worker and evaluator scopes.
  Args:
      var: tensorflow variable
  """
  if 'worker' in var.name.split('/')[0] or \
      'policy_eval' in var.name.split('/')[0]:
    return '/'.join(var.name.split('/')[1:])
  else:
    return var.name

def create_compositions(n, skills, worker=None):
  '''
  creates a composition network structure for policy and value networks
  according to the task. Always assuming compositions happen from left
  to right.
  Args:
      n: number of compositions in task
      skills: skills required in the task (indices) in the correct order
      worker: scope of worker requesting the graph, if None scope is assumed
        to be global
  '''
  # first all the skills involved
  skill_embedders = []
  for sk in skills:
    name_scope = 'global_{}'.format(sk)
    if worker:
      name_scope = worker + '/' + name_scope
    skill_embedders.append(Skill(
      name_scope=name_scope,
      state_dims=env_.get_state_size()))
  # now all the compositions from left to right
  policy_compositions = [skill_embedders[0]]
  value_compositions = [skill_embedders[0]]
  # starting from the first skill
  for i in range(n-1):
    next_embedding = skill_embedders[i+1]
    name_scope='compose_policy_{}'.format(i)
    if worker:
      name_scope = worker + '/' + name_scope
    policy_compositions.append(CompositionModule(
      name_scope=name_scope,
      embedder_1=policy_compositions[-1],
      embedder_2=next_embedding))
    name_scope='compose_value_{}'.format(i)
    if worker:
      name_scope = worker + '/' + name_scope
    value_compositions.append(CompositionModule(
      name_scope=name_scope,
      embedder_1=value_compositions[-1],
      embedder_2=next_embedding))
  # final layers sits on top of the last embedding
  policy_trainable_scopes = \
    ['compose_policy_{}/'.format(i) for i in range(n-1)]
  # scoping of variables depends on whether it is a worker graph or global
  if worker:
    policy_trainable_scopes = \
      [worker + '/' + name_scope for name_scope in policy_trainable_scopes]
  if worker:
    name_scope = worker
    global_final_layer = False
  else:
    global_final_layer = True
  policy_net = PolicyModule(
    name_scope=name_scope,
    trainable_scopes=policy_trainable_scopes,
    num_outputs=len(VALID_ACTIONS),
    embedder=policy_compositions[-1],
    global_final_layer=global_final_layer)
  value_trainable_scopes = \
    ['compose_value_{}/'.format(i) for i in range(n-1)]
  if worker:
    value_trainable_scopes = \
      [worker + '/' + name_scope for name_scope in value_trainable_scopes]
  if worker:
    name_scope = worker
  value_net = ValueModule(
    name_scope=name_scope,
    trainable_scopes=value_trainable_scopes,
    embedder=value_compositions[-1],
    global_final_layer=global_final_layer)
  return policy_net, value_net

# Depending on the game we may have a limited action space
env_ = make_env(FLAGS.env, FLAGS.task)
VALID_ACTIONS = list(range(env_.get_num_actions()))

# Set the number of workers
NUM_WORKERS = FLAGS.parallelism

# directories for logs and model files
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints", FLAGS.task)
LOG_DIR = os.path.join(MODEL_DIR, "logs", FLAGS.task)
PRIMITIVES_DIR = os.path.join(FLAGS.trained_primitives, "checkpoints", FLAGS.env)
TRANSFER_DIR = FLAGS.transfer_dir
NUM_SKILLS = len(FLAGS.task.split('_'))/2

SKILLS = []
if FLAGS.env == 'objects_env':
  splt = FLAGS.task.split('_')
  i = 0
  while i < len(splt):
    obj_ind = int(splt[i+1])
    if splt[i] == "collect" or \
        splt[i] == "or" or \
        splt[i] == "then":
      SKILLS.append(2*obj_ind)
    elif splt[i] == "evade":
      SKILLS.append(2*obj_ind+1)
    elif splt[i] == "and":
      SKILLS.append(2*obj_ind+1)
    else:
        raise ValueError("Unrecognized subtask {}".format(splt[i]))
    i += 2

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
  envs.append(make_env(FLAGS.env, FLAGS.task))

with tf.device("/cpu:0"):
  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Global policy and value nets for each task
  policy_net, value_net = create_compositions(NUM_SKILLS, SKILLS)

  # Global step iterator
  global_counter = itertools.count()

  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    # global scopes to copy weights from
    global_scopes = \
      ['global_{}'.format(sk) for sk in SKILLS] + \
      ['compose_policy_{}/'.format(i) for i in range(NUM_SKILLS)] + \
      ['policy_net/'] + \
      ['compose_value_{}/'.format(i) for i in range(NUM_SKILLS)] + \
      ['value_net/']
    # create the worker
    worker = Worker(
      name="worker_{}".format(worker_id),
      task_id=0,
      env=envs[worker_id],
      policy_net=policy_net,
      value_net=value_net,
      global_scopes=global_scopes,
      global_counter=global_counter,
      task_counter=global_counter,
      task_step=global_step,
      curriculum=None,
      max_global_steps=FLAGS.max_global_steps)
    # create local worker graph structure
    worker.policy_net, worker.value_net = create_compositions(
      NUM_SKILLS,
      SKILLS,
      "worker_{}".format(worker_id))
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
  global_scopes = \
    ['global_{}'.format(sk) for sk in SKILLS] + \
    ['compose_policy_{}/'.format(i) for i in range(NUM_SKILLS-1)] + \
    ['policy_net/'] + \
    ['compose_value_{}/'.format(i) for i in range(NUM_SKILLS)] + \
    ['value_net/']
  ev = PolicyEval(
    task_id=0,
    env=make_env(FLAGS.env, FLAGS.task),
    policy_net=policy_net,
    global_scopes=global_scopes,
    task_counter=global_counter,
    global_counter=global_counter,
    saver=saver,
    logfile=logfile,
    checkpoint_path=CHECKPOINT_DIR+'/')
  ev.policy_net, _ = create_compositions(NUM_SKILLS, SKILLS, "policy_eval_0")
  ev.create_copy_ops(key_func)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  # Load the primitives
  latest_checkpoint = tf.train.latest_checkpoint(PRIMITIVES_DIR)
  if latest_checkpoint:
    # only load these variables
    to_load = []
    for p in SKILLS:
      to_load += tf.contrib.slim.get_variables(
        scope='global_{}/'.format(p),
        collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    to_load += tf.contrib.slim.get_variables(
      scope='policy_net/'.format(p),
      collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    to_load += tf.contrib.slim.get_variables(
      scope='value_net/'.format(p),
      collection=tf.GraphKeys.TRAINABLE_VARIABLES)

    loader = tf.train.Saver(to_load)
    sys.stderr.write("\nLoading Primitives from: {}\n".format(latest_checkpoint))
    loader.restore(sess, latest_checkpoint)

  # Load the transfer compose layer if any
  if TRANSFER_DIR:
    latest_checkpoint = tf.train.latest_checkpoint(TRANSFER_DIR)
    if latest_checkpoint:
      # only load these variables
      to_load = []
      for p in range(NUM_SKILLS-1):
        to_load += tf.contrib.slim.get_variables(
          scope='compose_policy_{}/'.format(p),
          collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        to_load += tf.contrib.slim.get_variables(
          scope='compose_value_{}/'.format(p),
          collection=tf.GraphKeys.TRAINABLE_VARIABLES)
      loader = tf.train.Saver(to_load)
      sys.stderr.write("\nLoading composition layers from: {}\n".format(latest_checkpoint))
      loader.restore(sess, latest_checkpoint)
    # Start threads for policy eval task
  eval_threads = []
  converged_threads = itertools.count()
  eval_thread = threading.Thread(
    target=ev.continuous_eval,
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
