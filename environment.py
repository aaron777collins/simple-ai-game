from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from player import DO_NOTHING_ACTION
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import itertools

flatten = itertools.chain.from_iterable

tf.compat.v1.enable_v2_behavior()

MAX_GAMES = 3

def flatten_to_list(_2dlist):
        return list(flatten(_2dlist))

class Environment(py_environment.PyEnvironment):
    def __init__(self, game):
        self.game = game
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(flatten_to_list(self.game.visualization_data)),), dtype=np.float, minimum=0, maximum=1, name='observation'
        )
        self._game_count = 0

        self._episode_ended = False


    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game.reset()
        self._episode_ended = False
        return ts.restart(np.array(flatten_to_list(self.game.visualization_data), dtype=np.float))

    def _step(self, action):

        self.game.run(int(action))

        if self.game.lost:
            self._game_count+=1

            if(self._game_count >= MAX_GAMES):
                self._game_count = 0
                self._episode_ended = True
                return ts.termination(
                    np.array(flatten_to_list(self.game.visualization_data), dtype=np.float), reward=0.0
                )
            else:
                return self.reset()

        else:

            # if(self.game.player.rect.right > self.game.width/8*7
            # or self.game.player.rect.left < self.game.width/8
            # or self.game.player.rect.top < self.game.height/8
            # or self.game.player.rect.bottom > self.game.height/8*7):
            #     # Nullify rewards
            #     return ts.transition(
            #     np.array(flatten_to_list(self.game.visualization_data), dtype=np.float), reward=0.05, discount=1.0
            #     )

            if(self.game.passed_enemies):
                self.game.passed_enemies=False
                return ts.transition(
                np.array(flatten_to_list(self.game.visualization_data), dtype=np.float), reward=1, discount=1.0
                )


            return ts.transition(
                np.array(flatten_to_list(self.game.visualization_data), dtype=np.float), reward=0.5, discount=1.0
            )
