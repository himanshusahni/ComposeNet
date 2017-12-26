"""
@author: himanshusahni

Script for setting up environments for experiments
"""

import objects_env

def make_env(env, task):
  if env == 'objects_env':
    goal_arr = [0, 0, 0]
    splt = task.split('_')
    i = 0
    while i < len(splt):
      obj_ind = int(splt[i+1])
      if goal_arr[obj_ind] == 0:
        if splt[i] == "collect":
          goal_arr[obj_ind] = 1
        elif splt[i] == "or":
          goal_arr[obj_ind] = 1
        elif splt[i] == "then":
          try:
            goal_arr[obj_ind] = goal_arr[int(splt[i-1])] + 1
          except:
            raise ValueError("Incorrect task specification")
        elif splt[i] == "evade":
          goal_arr[obj_ind] = -1
        elif splt[i] == "and":
          try:
            if splt[i-2] == "evade":
              goal_arr[obj_ind] = -1
          except:
            raise ValueError("Incorrect task specification")
        else:
            raise ValueError("Unrecognized subtask {}".format(splt[i]))
      else:
        raise ValueError("Cannot re-specify same object twice")
      i += 2
    return objects_env.World(goal_arr=goal_arr)
  else:
    raise NotImplementedError("only objects environment supported right now")
