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
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
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
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--num-clones", type=int, default=3, help="number of clones in PSO")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50, help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000, help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    parser.add_argument("--param-noise-threshold", type=float, default=0.05, help="the desired KL divergence between perturbed and non-perturbed policy. set to < 0 to use a KL divergence relative to the eps-greedy exploration")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False, help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False, help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved.")
    parser.add_argument("--save-summary-dir", type=str, default="/tmp/atari", help="directory in which summary should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
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
    env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    if args.gym_monitor and savedir:
        env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    with U.make_session(24) as sess:
        with tf.device("/cpu:0"):
            # Create training graph and replay buffer
            def model_wrapper(img_in, num_actions, scope, **kwargs):
                actual_model = dueling_model if args.dueling else model
                return actual_model(img_in, num_actions, scope, layer_norm=args.layer_norm, **kwargs)
            act_f_clones, train_clones, update_target_clones, debug_clones, pso_update = deepq.build_train_PSO(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=model_wrapper,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                inertia=float(0.725),
                social=float(0.2),
                cognitive=float(0.2),
                num_clones=args.num_clones,
                gamma=0.99,
                grad_norm_clipping=10,
                double_q=args.double_q,
                param_noise=args.param_noise
            )

            # create the summary writer
            summary_writer = tf.summary.FileWriter(args.save_summary_dir, U.get_session().graph)

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
                replay_buffer = [ReplayBuffer(args.replay_buffer_size) for clone in range(args.num_clones)]

            U.initialize()
            for update_target in update_target_clones:
                update_target()

            num_iters = [0 for x in range(args.num_clones)]

            # Load the model
            state = maybe_load_model(savedir, container)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
                monitored_env.set_state(state["monitor_state"])

            start_time, start_steps = None, None
            steps_per_iter = RunningAvg(0.999)
            iteration_time_est = RunningAvg(0.999)
            obs = env.reset()
            num_iters_since_reset = 0
            reset = True
            global_counter = 0

            # Main trianing loop
            while True:

                for clone in range(args.num_clones):

                    while True:
                        global_counter += 1
                        num_iters[clone] += 1
                        num_iters_since_reset += 1
                        # Take action and store transition in the replay buffer.
                        kwargs = {}
                        if not args.param_noise:
                            update_eps = exploration.value(num_iters[clone])
                            update_param_noise_threshold = 0.
                        else:
                            if args.param_noise_reset_freq > 0 and num_iters_since_reset > args.param_noise_reset_freq:
                                # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                                reset = True

                            update_eps = 0.01  # ensures that we cannot get stuck completely
                            if args.param_noise_threshold >= 0.:
                                update_param_noise_threshold = args.param_noise_threshold
                            else:
                                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                                # for detailed explanation.
                                update_param_noise_threshold = -np.log(1. - exploration.value(num_iters[clone]) + exploration.value(num_iters[clone]) / float(env.action_space.n))
                            kwargs['reset'] = reset
                            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                            kwargs['update_param_noise_scale'] = (num_iters % args.param_noise_update_freq == 0)

                        action = act_f_clones[clone](np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                        reset = False
                        new_obs, rew, done, info = env.step(action)
                        replay_buffer[clone].add(obs, action, rew, new_obs, float(done))
                        obs = new_obs
                        if done:
                            num_iters_since_reset = 0
                            obs = env.reset()
                            reset = True

                        # if (num_iters[clone] > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                        #                 num_iters[clone] % args.learning_freq == 0):
                        #     # Sample a bunch of transitions from replay buffer
                        #     if args.prioritized:
                        #         experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters[clone]))
                        #         (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        #     else:
                        #         obses_t, actions, rewards, obses_tp1, dones = replay_buffer[clone].sample(args.batch_size)
                        #         weights = np.ones_like(rewards)
                        #     # Minimize the error in Bellman's equation and compute TD-error
                        #     td_errors = train_clones[0](obses_t, actions, rewards, obses_tp1, dones, weights)
                        #     # Update the priorities in the replay buffer
                        #     if args.prioritized:
                        #         new_priorities = np.abs(td_errors) + args.prioritized_eps
                        #         replay_buffer.update_priorities(batch_idxes, new_priorities)
                        # # Update target network.
                        # if num_iters[clone] % args.target_update_freq == 0:
                        #     update_target_clones[clone]()

                        if start_time is not None:
                            steps_per_iter.update(info['steps'] - start_steps)
                            iteration_time_est.update(time.time() - start_time)
                        start_time, start_steps = time.time(), info["steps"]

                        # Save the model and training state.
                        if num_iters[clone] > 0 and (num_iters[clone] % args.save_freq == 0 or info["steps"] > args.num_steps):
                            maybe_save_model(savedir, container, {
                                'replay_buffer': replay_buffer,
                                'num_iters': num_iters[clone],
                                'monitor_state': monitored_env.get_state(),
                            })

                        if info["steps"] > args.num_steps:
                            break

                        if done:

                            steps_left = args.num_steps - info["steps"]
                            completion = np.round(info["steps"] / args.num_steps, 1)

                            logger.record_tabular("% completion", completion)
                            logger.record_tabular("steps", info["steps"])
                            logger.record_tabular("iters", num_iters[clone])
                            logger.record_tabular("episodes", len(info["rewards"]))
                            logger.record_tabular("reward", np.mean(info["rewards"][-1:]))
                            logger.record_tabular("exploration", exploration.value(num_iters[clone]))
                            if args.prioritized:
                                logger.record_tabular("max priority", replay_buffer._max_priority)
                            fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                            if steps_per_iter._value is not None else "calculating...")
                            logger.dump_tabular()
                            logger.log()
                            logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                            logger.log()
                            break

                last_returns = [np.mean(info["rewards"][-args.num_clones*5 + clone::args.num_clones]) for clone in range(args.num_clones)]
                merged, group_best_return, clone_best_returns, group_best_q_func_vars = pso_update(last_returns)
                summary_writer.add_summary(merged, global_counter)
                logger.log("Last Reward {}".format(['%.2f' % return_ for return_ in last_returns]))
                logger.log("Clone Best Rewards {}".format(['%.2f' % return_ for return_ in clone_best_returns]))
