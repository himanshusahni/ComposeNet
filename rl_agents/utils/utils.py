import tensorflow as tf

'''
Network definition functions.
'''
def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    # return tf.Variable(initial)
    return tf.get_variable(name, shape=shape,
    	initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class Transition(object):

	def __init__(self, state, action, reward, next_state, terminal):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.terminal = terminal

def make_copy_params_op(v1_list, v2_list, key_func):
  """
  Creates an operation that copies parameters from variables in v1_list to
  variables in v2_list. The lists will be individually sorted by variable names.
  Args:
      v1_list: copy FROM list of tensorflow variables
      v2_list: copy TO list of tensorflow variables
  """
  v1_list = list(sorted(v1_list, key=key_func))
  v2_list = list(sorted(v2_list, key=key_func))
  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)
  return update_ops
