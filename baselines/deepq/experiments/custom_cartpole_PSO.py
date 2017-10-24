import argparse

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def arg_parse():
    parser = argparse.ArgumentParser("Cartpole PSO_DQN test")
    parser.add_argument("--num-clones", type=int, default=3, help="number of clones in PSO")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    with U.make_session(8):
        # Create the environment
        envs = [gym.make("CartPole-v0") for clone in range(args.num_clones)]
        # Create all the functions necessary to train the model
        act_f_clones, train_clones, update_target_clones, debug_clones, pso_update = deepq.build_train_PSO(
            make_obs_ph=lambda name: U.BatchInput(envs[0].observation_space.shape, name=name),
            q_func=model,
            num_actions=envs[0].action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            inertia=float(0.729),
            social=float(2.05),
            cognitive=float(2.05),
            num_clones=args.num_clones,
        )
        # Create the replay buffer
        replay_buffers = [ReplayBuffer(50000) for clone in range(args.num_clones)]
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        for update_target in update_target_clones:
            update_target()

        episode_rewards = [[0.0] for clone in range(args.num_clones)]
        obs = []
        for env in envs:
            obs.append(env.reset())

        for t in itertools.count():
            for clone in range(args.num_clones):
                # Take action and update exploration to the newest value
                action = act_f_clones[clone](obs[clone][None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = envs[clone].step(action)
                # Store transition in the replay buffer.
                replay_buffers[clone].add(obs[clone], action, rew, new_obs, float(done))
                obs[clone] = new_obs

                episode_rewards[clone][-1] += rew
                if done:
                    obs[clone] = envs[clone].reset()
                    episode_rewards[clone].append(0)

                is_solved = t > 100 and np.mean(episode_rewards[clone][-101:-1]) >= 200
                if is_solved:
                    # Show off the result
                    envs[clone].render()
                # else:
                #     # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                #     if t > 1000:
                #         obses_t, actions, rewards, obses_tp1, dones = replay_buffers[clone].sample(32)
                #         train_clones[clone](obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                #     # Update target network periodically.
                #     if t % 1000 == 0:
                #         update_target_clones[clone]()

                # if done and len(episode_rewards[clone]) % 10 == 0:
                #     logger.record_tabular("clone id", clone)
                #     logger.record_tabular("steps", t)
                #     logger.record_tabular("episodes", len(episode_rewards[clone]))
                #     logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[clone][-101:-1]), 1))
                #     logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                #     logger.dump_tabular()

            if t % 100 == 0 and t != 0:
                last_returns = [np.mean(episode_reward[-101:-1]) for episode_reward in episode_rewards]
                (pso_info,) = pso_update(last_returns)
                logger.log("Last Reward {}".format(['%.2f' % return_ for return_ in last_returns]))
                logger.log("Clone Best Rewards {}".format(['%.2f' % return_ for return_ in pso_info["clone_best"]]))
                # logger.dump_tabular()
