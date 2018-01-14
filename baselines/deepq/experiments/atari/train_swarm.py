import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import json

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import BootstrappedReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from baselines.deepq.experiments.atari.model import model, dueling_model


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8),
                        help="total number of steps to run the environment for")
    parser.add_argument("--num-clones", type=int, default=3, help="number of clones in PSO")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000,
                        help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50,
                        help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000,
                        help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    parser.add_argument("--param-noise-threshold", type=float, default=0.05,
                        help="the desired KL divergence between perturbed and non-perturbed policy. set to < 0 to use a KL divergence relative to the eps-greedy exploration")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False,
                 help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False,
                 help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False,
                 help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4,
                        help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None,
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-summary-dir", type=str, default="/tmp/atari",
                        help="directory in which summary should be saved.")
    parser.add_argument("--save-summary-freq", type=int, default=2000,
                        help="directory in which summary should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6,
                        help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True,
                 help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(
        monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    args = parse_args()

    # Parse savedir and azure container.
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
    # Create and seed the env.
    envs = []
    monitored_envs = []
    for clone in range(args.num_clones):
        env, monitored_env = make_env(args.env)
        envs.append(env)
        monitored_envs.append(monitored_env)

    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    if args.gym_monitor and savedir:
        env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    with U.make_session(24) as sess:
        # Create training graph and replay buffer
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            actual_model = dueling_model if args.dueling else model
            return actual_model(img_in, num_actions, scope, layer_norm=args.layer_norm, **kwargs)


        act, train, update_target = deepq.build_graph_swarm.build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=model_wrapper,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            num_clones=args.num_clones,
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise
        )

        # create the summary writer
        if args.save_summary_dir:
            summary_writer = tf.summary.FileWriter(args.save_summary_dir, U.get_session().graph)
            episode_rewards_ph = tf.placeholder(tf.float64, (args.num_clones))
            last_returns_summary_op = tf.summary.merge(
                [tf.summary.scalar("debug_returns_clone" + str(i), tf.reduce_mean(episode_rewards_ph[i])) for i in
                 range(args.num_clones)])

        approximate_num_iters = args.num_steps / 40
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = BootstrappedReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = [None] * args.num_clones, [None] * args.num_clones
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = []
        for env in envs:
            obs.append(env.reset())

        num_iters_since_reset = [0 for clone in range(args.num_clones)]
        last_returns = [None] * args.num_clones
        reset = True

        # Main training loop
        while True:
            num_iters += 1

            for clone in range(args.num_clones):
                num_iters_since_reset[clone] += 1
                # Take action and store transition in the replay buffer.
                kwargs = {}
                if not args.param_noise:
                    update_eps = exploration.value(num_iters)
                    update_param_noise_threshold = 0.
                else:
                    if args.param_noise_reset_freq > 0 and num_iters_since_reset > args.param_noise_reset_freq:
                        # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                        reset = True

                    update_eps = 0.01  # ensures that we cannot get stuck completely
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1. - exploration.value(num_iters) + exploration.value(num_iters) / float(env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = (num_iters % args.param_noise_update_freq == 0)

                action, cover_agents, summary_act = act(np.array(obs[clone])[None], clone, update_eps=update_eps,
                                                        **kwargs)
                reset = False
                new_obs, rew, done, info = envs[clone].step(action[0])
                replay_buffer.add(obs[clone], action[0], rew, new_obs, float(done), cover_agents[0])
                obs[clone] = new_obs

                if start_time[clone] is not None:
                    steps_per_iter.update(info['steps'] - start_steps[clone])
                    iteration_time_est.update(time.time() - start_time[clone])
                start_time[clone], start_steps[clone] = time.time(), info["steps"]

                if done:
                    num_iters_since_reset[clone] = 0
                    obs[clone] = envs[clone].reset()
                    reset = True

                    last_returns[clone] = np.mean(info["rewards"][-5:])

                    steps_left = args.num_steps - info["steps"]
                    completion = np.round(info["steps"] / args.num_steps, 1)

                    logger.record_tabular("clone id", clone)
                    logger.record_tabular("% completion", completion)
                    logger.record_tabular("steps", info["steps"])
                    logger.record_tabular("iters", num_iters)
                    logger.record_tabular("episodes", len(info["rewards"]))
                    logger.record_tabular("reward (5 epi mean)", np.mean(info["rewards"][-5:]))
                    logger.record_tabular("exploration", exploration.value(num_iters))
                    if args.prioritized:
                        logger.record_tabular("max priority", replay_buffer._max_priority)
                    fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                    if steps_per_iter._value is not None else "calculating...")
                    logger.dump_tabular()
                    logger.log()
                    logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                    logger.log()

            if num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and num_iters % args.learning_freq == 0:
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones, shares = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors, summary_train = train(obses_t, actions, rewards, obses_tp1, dones, np.transpose(shares),
                                                 weights)

                if args.save_summary_dir:
                    summary_writer.add_summary(summary_train, num_iters)

                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()
                logger.log("Last Reward {}".format(['%.2f' % return_ for return_ in last_returns]))

            if num_iters % args.save_summary_freq == 0 and num_iters != 0:
                if args.save_summary_dir:
                    last_returns_summary = U.get_session().run(last_returns_summary_op,
                                                               {episode_rewards_ph: last_returns})
                    summary_writer.add_summary(last_returns_summary, num_iters)

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env[clone].get_state(),
                })

            if num_iters > args.num_steps:
                break
