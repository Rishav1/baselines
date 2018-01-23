import argparse

import itertools
from mdp_environment import ActionGenerator, StateGenerator, MDPModel
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def linear_mdp(num_nodes, state_size):
    states = []
    actions = []
    action_generator = ActionGenerator('name')
    actions.append(action_generator.generate_action(name='LEFT'))
    actions.append(action_generator.generate_action(name='RIGHT'))
    state_generator = StateGenerator('name', 'value', 'reward')
    for i in range(num_nodes):
        if i == 0:
            reward = 1/num_nodes
        elif i == (num_nodes-1):
            reward = 1
        else:
            reward = 0
        states.append(state_generator.generate_state(name=i, value=np.random.random_sample(state_size), reward=reward))

    linear_model = MDPModel('Linear')
    linear_model.add_actions(actions)
    linear_model.add_states(states)
    for i in range(num_nodes - 2):
        linear_model.add_transition(states[i + 1], actions[0], {states[i]: 1})
        linear_model.add_transition(states[i + 1], actions[1], {states[i + 2]: 1})

    linear_model.add_init_states({next(linear_model.get_states(1)): 1})
    linear_model.finalize()
    linear_model.visualize()
    return linear_model


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def arg_parse():
    parser = argparse.ArgumentParser("MDP DQN test")
    parser.add_argument("--save-summary-dir", type=str, default="", help="default tensorboard summary directory")
    parser.add_argument("--mdp-arity", type=int, default=1, help="arity of MDP")
    parser.add_argument("--mdp-dimension", type=int, default=10, help="size of MDP")
    parser.add_argument("--mdp-state-size", type=int, default=5, help="representational dimension of MDP states")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    with U.make_session(8):
        # Create the environment
        if args.mdp_arity == 1:
            mdp = linear_mdp(args.mdp_dimension, args.mdp_state_size)
        else:
            mdp = None
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_graph.build_train(
            make_obs_ph=lambda name: U.BatchInput((args.mdp_state_size, ), name=name),
            q_func=model,
            num_actions=2**args.mdp_arity,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=1000, initial_p=1, final_p=0.02)

        # create the summary writer
        if args.save_summary_dir:
            summary_writer = tf.summary.FileWriter(args.save_summary_dir, U.get_session().graph)
            episode_rewards_ph = tf.placeholder(tf.float64, ())
            last_returns_summary_op = tf.summary.scalar("debug_returns", tf.reduce_mean(episode_rewards_ph))

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        obs = mdp.initialize()
        episode_rewards = [obs.reward]
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action_id = act(obs.value[None], update_eps=exploration.value(t))[0]
            new_obs = mdp.transition(next(mdp.get_actions(action_id)))
            # Store transition in the replay buffer.
            replay_buffer.add(obs.value, action_id, new_obs.reward, new_obs.value, float(mdp.is_terminated()))
            obs = new_obs

            episode_rewards[-1] += new_obs.reward
            if mdp.is_terminated() or t % 2000 == 0:
                obs = mdp.initialize()
                episode_rewards.append(obs.reward)

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 1000:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(128)
                td_error, summary = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if args.save_summary_dir:
                    summary_writer.add_summary(summary, t)

            # Update target network periodically.
            if t % 1000 == 0 and t != 0:
                update_target()
                last_returns = np.mean(episode_rewards[-50:-1])
                # summary_writer.add_summary(merged, t)
                logger.log("Last Reward {}".format('%.2f' % last_returns))

            # Add mean returns to the summary
            if args.save_summary_dir and t % 50 == 0:
                last_returns_summary = U.get_session().run(last_returns_summary_op, {episode_rewards_ph: last_returns})
                summary_writer.add_summary(last_returns_summary, t)
