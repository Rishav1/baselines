import argparse

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import BootstrappedReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def arg_parse():
    parser = argparse.ArgumentParser("Cartpole Swarm DQN test")
    parser.add_argument("--num-clones", type=int, default=3, help="number of clones in swarm")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    with U.make_session(8):
        # Create the environment
        envs = [gym.make("CartPole-v0") for clone in range(args.num_clones)]
        # Create all the functions necessary to train the model
        act_f, train, update_target = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(envs[0].observation_space.shape, name=name),
            q_func=model,
            num_actions=envs[0].action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            num_clones=args.num_clones,
        )
        # Create the replay buffer
        replay_buffers = BootstrappedReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=1000, initial_p=1, final_p=0.02)

        #create the summary writer
        summary_writer = tf.summary.FileWriter('/tmp/train', U.get_session().graph)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [[0.0] for clone in range(args.num_clones)]
        obs = []
        for env in envs:
            obs.append(env.reset())

        for t in itertools.count():
            for clone in range(args.num_clones):
                # Take action and update exploration to the newest value
                action, cover_agents = act_f(obs[clone][None], clone, update_eps=exploration.value(t))
                new_obs, rew, done, _ = envs[clone].step(action[0])
                # Store transition in the replay buffer.
                replay_buffers.add(obs[clone], action[0], rew, new_obs, float(done), cover_agents[0])
                obs[clone] = new_obs

                episode_rewards[clone][-1] += rew
                if done:
                    obs[clone] = envs[clone].reset()
                    episode_rewards[clone].append(0)

                is_solved = t > 100 and np.mean(episode_rewards[clone][-101:-1]) >= 200

                if is_solved:
                    envs[clone].render()
                else:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones, shares = replay_buffers.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.transpose(shares), np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                # if done and len(episode_rewards[clone]) % 10 == 0:
                #     logger.record_tabular("clone id", clone)
                #     logger.record_tabular("steps", t)
                #     logger.record_tabular("episodes", len(episode_rewards[clone]))
                #     logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[clone][-5:-1]), 1))
                #     logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                #     logger.dump_tabular()

            if t % 1000 == 0 and t != 0:
                last_returns = [np.mean(episode_reward[-5:-1]) for episode_reward in episode_rewards]
                # summary_writer.add_summary(merged, t)
                logger.log("Last Reward {}".format(['%.2f' % return_ for return_ in last_returns]))
                # logger.log("Clone Best Rewards {}".format(['%.2f' % return_ for return_ in clone_best_returns]))
