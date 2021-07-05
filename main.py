from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import environ
import os
from player import DO_NOTHING_ACTION

from pygame import time

from depencencies import installAll
# Installs dependencies if they haven't been installed yet
installAll() 

from game import Game
from environment import Environment

import abc
import tensorflow as tf
import numpy as np

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


# IMPORTANT PARAMETER
learning_rate = 0.0001

checkpoint_dir = os.path.join(__file__, 'checkpoint')
policy_dir = os.path.join(__file__, 'policy')

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

if __name__ == "__main__":
    global gameInstance
    gameInstance_train = Game(1280, 720, 60, "Simple Ai Game", int(1280/10), int(720/10), 100, 10, 1, display_visualization=True, slow_down_game=False, speed=5)

    gameInstance_eval = Game(1280, 720, 60, "Simple Ai Game", int(1280/10), int(720/10), 100, 10, 1, display_visualization=True, slow_down_game=False, speed=5)

    environment_train = tf_py_environment.TFPyEnvironment(Environment(gameInstance_train))
    environment_eval = tf_py_environment.TFPyEnvironment(Environment(gameInstance_eval))

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
    fc_layer_params = (100, 128)
    action_tensor_spec = tensor_spec.from_spec(environment_train.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [ dense_layer(num_units) for num_units in fc_layer_params]

    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2)
    )

    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        environment_train.time_step_spec(),
        environment_train.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(environment_train.time_step_spec(), environment_train.action_spec())

    time_step = environment_train.reset()

    random_policy.action(time_step)


    # print(compute_avg_return(environment_eval, random_policy, 3))


# ######### replay buffer settings ############## 

    replay_buffer_max_length = 100000

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=environment_train.batch_size,
        max_length=replay_buffer_max_length
    )

    # initial_collect_steps = 5

    # collect_data(environment_train, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=environment_train.batch_size, 
    num_steps=2).prefetch(3)

    iterator = iter(dataset)
    print(iterator)

    # #############################################

    # Added settings
    num_eval_episodes = 3
    num_iterations = 100
    collect_steps_per_iteration = 500
    log_interval = 1
    eval_interval = 5

    # try:
    #     # %%time
    # except:
    #     pass

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(environment_train, agent.policy, num_eval_episodes)
    returns = [avg_return]

    max_return = 0

    # # Try to restore from checkpoint

    # train_checkpointer = common.Checkpointer(
    # ckpt_dir=checkpoint_dir,
    # max_to_keep=1,
    # agent=agent,
    # policy=agent.policy,
    # replay_buffer=replay_buffer,
    # global_step=agent.train_step_counter.numpy()
    # )
    
    # tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    # try:
    #     train_checkpointer.initialize_or_restore()
    #     tf_policy_saver = tf.compat.v2.saved_model.load(policy_dir)
    # except:
    #     print("Error restoring from checkpoint")


    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(environment_train, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(environment_eval, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            max_return = (max_return, avg_return)[avg_return>max_return]


    environment_train.close()
    environment_eval.close()
        
    iterations = range(0, num_iterations + 1, eval_interval)
    print(iterations)
    print(returns)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=max_return)
    plt.show()



    # Save AI

    # train_checkpointer.save(agent.train_step_counter.numpy())

    # tf_policy_saver.save(policy_dir)



    # ##############################################