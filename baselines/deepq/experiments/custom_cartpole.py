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
    parser = argparse.ArgumentParser("Cartpole Swarm DQN test")
    parser.add_argument("--save-summary-dir", type=str, default="", help="default tensorboard summary directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_graph.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=1000, initial_p=1, final_p=0.02)

        #create the summary writer
        if args.save_summary_dir:
            summary_writer = tf.summary.FileWriter(args.save_summary_dir, U.get_session().graph)
            episode_rewards_ph = tf.placeholder(tf.float64, ())
            last_returns_summary_op = tf.summary.scalar("debug_returns", tf.reduce_mean(episode_rewards_ph))

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    td_error, summary = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    if args.save_summary_dir:
                        summary_writer.add_summary(summary, t)

            # Add mean returns to the summary
            if args.save_summary_dir and t % 50 == 0:
                last_returns = np.mean(episode_rewards[-5:-1])
                last_returns_summary = U.get_session().run(last_returns_summary_op, {episode_rewards_ph: last_returns})
                summary_writer.add_summary(last_returns_summary, t)

            # Update target network periodically.
            if t % 1000 == 0 and t != 0:
                update_target()
                # summary_writer.add_summary(merged, t)
                logger.log("Last Reward {}".format('%.2f' % last_returns))
                # logger.log("Clone Best Rewards {}".format(['%.2f' % return_ for return_ in clone_best_returns]))

            # if done and len(episode_rewards) % 10 == 0:
            #     logger.record_tabular("steps", t)
            #     logger.record_tabular("episodes", len(episode_rewards))
            #     logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            #     logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            #     logger.dump_tabular()
