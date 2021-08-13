from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from helper import convert_list_to_dash_separated
from os import environ
import os
from player import DO_NOTHING_ACTION

from pygame import time

from game import Game
from environment import Environment

import abc
import tensorflow as tf
import numpy as np

import shutil

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'
        )
    )



def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see tutorial 4 or the drivers module.
# https://github.com/tensorflow/agents/blob/master/docs/tutorials/4_drivers_tutorial.ipynb 
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers


#######################################################################################

def delete_progess(trainer):
  try:
    shutil.rmtree(trainer.policy_dir)
  except:
    pass

class Trainer:


    def __init__(self, 
      num_eval_episodes = 2,
      num_iterations = 20,
      collect_steps_per_iteration = 100,
      log_interval = 1,
      eval_interval = 3,
      show_chart = False,
      learning_rate=0.00005,
      window_scale=1.0,
      display_visualization=False,
      fc_layer_params = (922, 256, 128)):

      self.display_visualization=display_visualization

      self.window_scale=window_scale

      self.gameInstance_train = Game(1280, 720, 60, "Simple Ai Game", int(1280/10), int(720/10), 100, 10, 1, display_visualization=display_visualization, slow_down_game=False, speed=5, window_scale=self.window_scale)

      self.gameInstance_eval = Game(1280, 720, 60, "Simple Ai Game", int(1280/10), int(720/10), 100, 10, 1, display_visualization=display_visualization, slow_down_game=False, speed=5, window_scale=self.window_scale)

      self.environment_train = tf_py_environment.TFPyEnvironment(Environment(self.gameInstance_train))
      self.environment_eval = tf_py_environment.TFPyEnvironment(Environment(self.gameInstance_eval))

      # Added settings
      self.num_eval_episodes = num_eval_episodes
      self.num_iterations = num_iterations
      self.collect_steps_per_iteration = collect_steps_per_iteration
      self.log_interval = log_interval
      self.eval_interval = eval_interval
      self.show_chart = show_chart

      # IMPORTANT PARAMETERS
      # learning_rate = 0.00005
      self.learning_rate = learning_rate

      # utils.validate_py_environment(Environment(gameInstance_train), episodes=5)
      # print(isinstance(environment_train, tf_environment.TFEnvironment))
      # print("TimeStep Specs:", environment_train.time_step_spec())
      # print("Action Specs:", environment_train.action_spec())

      # time_step = environment_train.reset()
      # print('Time step:')
      # print(time_step)

      # action = np.array(3, dtype=np.int32)

      # next_time_step = environment_train.step(action)
      # print('Next time step:')
      # print(next_time_step)

      # fc_layer_params = (100, 50)
      # self.fc_layer_params = (100, 128)
      # self.fc_layer_params = (128, 128, 128)
      # self.fc_layer_params = (256, 128, 64, 32)
      # self.fc_layer_params = (9216, 128, 72)
      self.fc_layer_params = fc_layer_params

      fc_layer_params_str = "-" + str(learning_rate) + "-"

      fc_layer_params_str = fc_layer_params_str + "["


      fc_layer_params_str = fc_layer_params_str + convert_list_to_dash_separated(self.fc_layer_params)

      fc_layer_params_str = fc_layer_params_str + "]"


      # IMPORTANT CONSTANT
      self.policy_dir = os.path.join(os.getcwd(), 'policies', fc_layer_params_str)

      self.action_tensor_spec = tensor_spec.from_spec(self.environment_train.action_spec())
      self.num_actions = self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1

      self.dense_layers = [ dense_layer(num_units) for num_units in self.fc_layer_params]

      self.q_values_layer = tf.keras.layers.Dense(
          self.num_actions,
          activation=None,
          kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
          bias_initializer=tf.keras.initializers.Constant(-0.2)
      )

      self.q_net = sequential.Sequential(self.dense_layers + [self.q_values_layer])

      self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

      self.train_step_counter = tf.Variable(0)

      self.agent = dqn_agent.DqnAgent(
          self.environment_train.time_step_spec(),
          self.environment_train.action_spec(),
          q_network=self.q_net,
          optimizer=self.optimizer,
          td_errors_loss_fn=common.element_wise_squared_loss,
          train_step_counter=self.train_step_counter
      )

      self.agent.initialize()

      self.eval_policy = self.agent.policy
      self.collect_policy = self.agent.collect_policy


      # random_policy = random_tf_policy.RandomTFPolicy(environment_train.time_step_spec(), environment_train.action_spec())


      # time_step = environment_train.reset()

      # random_policy.action(time_step)


      # print(compute_avg_return(environment_eval, random_policy, 3))


  # ######### replay buffer settings ############## 

      self.replay_buffer_max_length = 100000

      self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          data_spec=self.agent.collect_data_spec,
          batch_size=self.environment_train.batch_size,
          max_length=self.replay_buffer_max_length
      )

      # initial_collect_steps = 5

      # collect_data(environment_train, random_policy, replay_buffer, initial_collect_steps)

      self.dataset = self.replay_buffer.as_dataset(
      num_parallel_calls=3, 
      sample_batch_size=self.environment_train.batch_size, 
      num_steps=2).prefetch(3)

      self.iterator = iter(self.dataset)
      print(self.iterator)

      # #############################################

    # try:
    #     # %%time
    # except:
    #     pass

    def cleanup(self):
      self.environment_train.close()
      self.environment_eval.close()

    def train_iteration(self):


       # Initialize policy saver
      self.tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)

      try:
        self.saved_policy = tf.compat.v2.saved_model.load(self.policy_dir)
        print(f"Found a policy at {self.policy_dir}")
      except:
        print("No policy found! Generating random policy instead.")

      # (Optional) Optimize by wrapping some of the code in a graph using TF function.
      self.agent.train = common.function(self.agent.train)

      # Reset the train step
      self.agent.train_step_counter.assign(0)

      # Evaluate the agent's policy once before training.
      self.avg_return = compute_avg_return(self.environment_train, self.agent.policy, self.num_eval_episodes)
      self.returns = [self.avg_return]

      self.max_return = 0

      # Note the loss
      self.losses = [0]
      self.max_loss = 0


      for _ in range(self.num_iterations):

          # Collect a few steps using collect_policy and save to the replay buffer.
          collect_data(self.environment_train, self.agent.collect_policy, self.replay_buffer, self.collect_steps_per_iteration)

          # Sample a batch of data from the buffer and update the agent's network.
          self.experience, unused_info = next(self.iterator)
          self.train_loss = self.agent.train(self.experience).loss

          self.step = self.agent.train_step_counter.numpy()

          if self.step % self.log_interval == 0:
              print('step = {0}: loss = {1}'.format(self.step, self.train_loss))

          if self.step % self.eval_interval == 0:
              self.avg_return = compute_avg_return(self.environment_eval, self.agent.policy, self.num_eval_episodes)
              print('step = {0}: Average Return = {1}'.format(self.step, self.avg_return))
              self.returns.append(self.avg_return)
              self.max_return = (self.max_return, self.avg_return)[self.avg_return>self.max_return]
              self.losses.append(self.train_loss)
              self.max_loss = (self.max_loss, self.train_loss)[self.train_loss>self.max_loss]



          
      self.iterations = range(0, self.num_iterations + 1, self.eval_interval)
      print(self.iterations)
      print(self.returns)

      if self.show_chart:
        plt.plot(self.iterations, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=self.max_return)
        plt.show()

        plt.plot(self.iterations, self.losses)
        plt.ylabel('Average Loss')
        plt.xlabel('Iterations')
        plt.ylim(top=self.max_loss)
        plt.show()

      # Saving policy
      print("Saved policy to " + self.policy_dir)
      self.tf_policy_saver.save(self.policy_dir)

      # ##############################################