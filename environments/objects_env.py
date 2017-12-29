"""
@author: himanshusahni

Simple object/enemy environment for toy experiments.
"""

import math
import numpy as np
import random
from copy import deepcopy


class World():

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    AGENT_COLOR = 10
    EMPTY_COLOR = 0
    STEP_COST = -0.01
    GRID_SIZE = 25
    DILATION_FACTOR = 3
    NUM_ACTIONS = 4
    NUM_OBJECTS = 3

    def __init__(
            self, object_pos=None, goal_arr=[-1, -1, 0]):
        """Initializes the environment. Multiple objects maybe targeted
        (rewards follow OR) or avoided (rewards follow AND).

        Args:
            goal_arr: Array indicating index of object to target and/or index
                      of object to avoid. +1 means target -1 means avoid.
        """
        self.orig_goal_arr = deepcopy(goal_arr)
        self.reset(object_pos)
        self.target_exists = any([g>0 for g in goal_arr])

        assert math.sqrt(self.GRID_SIZE) - int(math.sqrt(self.GRID_SIZE)) == 0

    def reset(self, object_pos=None, max_steps=10):
        self.max_steps = max_steps
        self.goal_arr = deepcopy(self.orig_goal_arr)
        self.visible = [True]*self.NUM_OBJECTS
        self.side_len = int(math.sqrt(self.GRID_SIZE))
        if not object_pos:
            # if object positions not manually specified, drawn randomly
            object_pos_flat = np.random.choice(
                self.GRID_SIZE, self.NUM_OBJECTS, replace=False)
            object_pos_2d = np.unravel_index(
                object_pos_flat, (self.side_len, self.side_len))
            self.object_positions_2d = [[object_pos_2d[0][i], object_pos_2d[1][i]] \
                for i in xrange(self.NUM_OBJECTS)]
        else:
            self.object_positions_2d = deepcopy(object_pos)

        self.object_dict = {(i + 1) * (255 / float(self.NUM_OBJECTS)) : i \
            for i in xrange(self.NUM_OBJECTS)}
        self.object_dict_inverted = {y:x \
            for x,y in self.object_dict.iteritems()}

        self.world = self.generate_map()
        self.image = self.construct_image()

        self.num_steps = 0

        return np.expand_dims(self.image.copy(), axis=-1)

    def reset_test(self, object_pos=None, goal_arr=[-1, 0, 1]):
        """Version of reset that returns both the dilated image and the original world."""
        self.reset()
        return np.expand_dims(self.world.copy(), axis=-1), np.expand_dims(
            self.image.copy(), axis=-1)

    def get_state_size(self):
        return [np.shape(self.image)[0], np.shape(self.image)[1]]

    def get_num_actions(self):
        return self.NUM_ACTIONS

    def generate_map(self):

        world = self.EMPTY_COLOR * np.ones((self.side_len, self.side_len))

        for i in xrange(len(self.object_positions_2d)):
            x,y = self.object_positions_2d[i]
            world[x,y] = self.object_dict_inverted[i]

        # Agent starting position.
        agent_pos = (0,0)  # random.randint(0, self.side_len - 1)

        # Agent cannot start at an object position.
        while world[agent_pos] != self.EMPTY_COLOR:
            agent_pos = (random.randint(0, self.side_len - 1),
                random.randint(0, self.side_len - 1))

        self.x = agent_pos[1] % self.side_len
        self.y = agent_pos[0] / self.side_len

        world[self.y][self.x] = self.AGENT_COLOR

        return world

    def construct_image(self):
        small_img = self.world.copy()
        for i in xrange(len(self.object_positions_2d)):
            # if the object is invisible and nothing else in on top
            [y,x] = self.object_positions_2d[i]
            if not self.visible[i] and \
                    self.world[y,x] == self.object_dict_inverted[i]:
                small_img[y,x] = self.EMPTY_COLOR
        row_dilated = np.repeat(small_img, self.DILATION_FACTOR, axis=0)
        col_dilated = np.repeat(row_dilated, self.DILATION_FACTOR, axis=1)
        return col_dilated

    def step(self, action):
        self.move_agent(action)
        self.move_avoid_objects()

        self.num_steps += 1

        reward = self.reward()
        done = self.isTerminal()
        action_list = ['up', 'down', 'left', 'right']

        self.image = self.construct_image()

        return np.expand_dims(self.image.copy(), axis=-1), reward, done

    def move_avoid_objects(self):
        """Moves all the avoid objects on the map towards the agent."""
        avoid_objs = [ind for ind in range(self.NUM_OBJECTS) \
            if self.goal_arr[ind] == -1]
        new_object_pos = [None]*self.NUM_OBJECTS
        for avoid_object_index in avoid_objs:
            obj = self.object_positions_2d[avoid_object_index]
            old_x = obj[1]
            old_y = obj[0]

            delta_x = 0
            delta_y = 0

            # Determine which delta x direction to move the avoid object.
            if old_x < self.x:
                delta_x = 1
            elif old_x > self.x:
                delta_x = -1

            if old_y < self.y:
                delta_y = 1
            elif old_y > self.y:
                delta_y = -1

            if delta_x and delta_y:
                # Whether to move the object in the x or y direction.
                move_val = random.randint(0, 1)

                if not move_val:
                    delta_y = 0
                else:
                    delta_x = 0

            new_x = old_x + delta_x
            new_y = old_y + delta_y

            # Avoid object teleports across the borders.
            new_x = new_x % self.side_len
            new_y = new_y % self.side_len

            if [new_y, new_x] in new_object_pos:
                new_x, new_y = old_x, old_y

            # Avoid object could have been on top of stationary object,
            # in which case the stationary object should re-appear when
            # the avoid object moves off of that location.
            new_object_pos[avoid_object_index] = [new_y, new_x]

        for obj_ind in range(self.NUM_OBJECTS):
            if self.goal_arr[obj_ind] >= 0:
                new_object_pos[obj_ind] = self.object_positions_2d[obj_ind]

        # empty out the old object positions
        for obj_ind in range(self.NUM_OBJECTS):
            [y,x] = self.object_positions_2d[obj_ind]
            if not self.world[y,x] == self.AGENT_COLOR:
                self.world[y,x] = self.EMPTY_COLOR
        # draw back in the target and neutral objects
        for obj_ind in range(self.NUM_OBJECTS):
            if self.goal_arr[obj_ind] >= 0:
                [y,x] = new_object_pos[obj_ind]
                if not self.world[y,x] == self.AGENT_COLOR:
                    self.world[y,x] = self.object_dict_inverted[obj_ind]
        # now draw in avoid objects (so they are always on top)
        for obj_ind in range(self.NUM_OBJECTS):
            if self.goal_arr[obj_ind] == -1:
                [y,x] = new_object_pos[obj_ind]
                if not self.world[y,x] == self.AGENT_COLOR:
                    self.world[y,x] = self.object_dict_inverted[obj_ind]

        self.object_positions_2d = new_object_pos


    def move_agent(self, action):
        new_x = self.x
        new_y = self.y

        if action == self.UP:
            new_y = self.y - 1
        if action == self.DOWN:
            new_y = self.y + 1
        if action == self.LEFT:
            new_x = self.x - 1
        if action == self.RIGHT:
            new_x = self.x + 1

        new_x = new_x % self.side_len
        new_y = new_y % self.side_len

        # Agent could have been on top of stationary object,
        # in which case the stationary object should re-appear when
        # the agent moves off of that location.
        try:
            obj_ind = self.object_positions_2d.index([self.y,self.x])
        except ValueError:
            obj_ind = -1
        if obj_ind >= 0:
            self.world[self.y][self.x] = self.object_dict_inverted[obj_ind]
        else:
            self.world[self.y][self.x] = self.EMPTY_COLOR

        self.x = new_x
        self.y = new_y

        obj_inds = [i for i, obj in enumerate(self.object_positions_2d) \
            if obj == [self.y, self.x]]
        for ind in obj_inds:
            # if agent steps on a target object, make it invisible
            if self.goal_arr[ind] > 0:
                self.visible[ind] = False
            # if agent steps on a non-immediate goal, make the game unwinnable
            if self.goal_arr[ind] > 1:
                self.goal_arr[ind] = float("inf")

        self.world[self.y][self.x] = self.AGENT_COLOR

    def step_test(self, action):
        """Version of step that returns both the dilated image and the original world."""
        image, reward, done = self.step(action)
        return np.expand_dims(self.world.copy(), axis=-1), np.expand_dims(
            self.image.copy(), axis=-1), reward, done

    def reward(self):
        # timeout!
        if self.num_steps >= self.max_steps:
            if self.target_exists:
                return -1
            # otherwise bring total reward to 1
            else:
                return 1 + self.STEP_COST*(self.max_steps-1)
        obj_inds = [i for i, obj in enumerate(self.object_positions_2d) \
            if obj == [self.y, self.x]]
        # agent is on top of an object(s)
        if len(obj_inds) > 0:
            # if on top of any target object
            if any([self.goal_arr[ind] == 1 for ind in obj_inds]):
                # reduce goal priority of every target object by 1
                self.goal_arr = [g-1 if g > 0 else g for g in self.goal_arr]
                if any([g>0 for g in self.goal_arr]):
                    # if any other goals left, don't reward yet
                    return self.STEP_COST
                else:
                    # else episode is over
                    return 1
            # if on top of any avoid object
            elif any([self.goal_arr[ind] == -1 for ind in obj_inds]):
                return -1
            # if on top of neutral object
            else:
                if self.target_exists:
                    return self.STEP_COST
                else:
                    return -self.STEP_COST
        # agent on empty location
        else:
            # there is a target object
            if self.target_exists:
                #the agent should try to reach the target
                # object as fast as possible, so it receives a negative
                # step cost at each time step.
                return self.STEP_COST
            # only avoid objects in the map
            else:
                # the agent only has to outrun the enemies
                # so it receives a small positive reward at each time step.
                return -self.STEP_COST

    def isTerminal(self):
        # timeout!
        if self.num_steps >= self.max_steps:
            return True

        obj_inds = [i for i, obj in enumerate(self.object_positions_2d) \
            if obj == [self.y, self.x]]
        # agent is on top of an object(s)
        if len(obj_inds) > 0:
            # it is an avoid object
            if any([self.goal_arr[ind] == -1 for ind in obj_inds]):
                return True
            if self.target_exists:
                # this means all targets have been achieved
                if all([g <= 0 for g in self.goal_arr]):
                    return True

        return False
