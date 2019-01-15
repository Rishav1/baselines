from baselines.common.input import observation_input
from baselines.common.tf_util import adjust_shape
import tensorflow as tf

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplementedError

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space

        Parameters
        ----------

        observation_space:
                observation space of the environment. Should be one of the gym.spaces types
        name: str
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt

# ================================================================
# Action slector functions 
# ================================================================

def ccav_aggregator(action_set):
    """Find a family of subsets that covers the universal set"""

    batch_size = tf.shape(action_set)[0]
    num_agents = tf.shape(action_set)[1]    # num_agents
    num_actions = tf.shape(action_set)[2]    # num_actions

    def body(cover):
        universe_extended = tf.logical_not(tf.tile(tf.matmul(action_set, tf.expand_dims(cover, 2)), [1, 1, num_actions]) > 0)
        s = tf.reduce_sum(tf.multiply(action_set, tf.cast(universe_extended, dtype=tf.int32)), axis=1)
        c = tf.one_hot(tf.argmax(s, axis=1), num_actions, dtype=tf.int32)
        cover = cover + c * (1 - cover)
        return cover

    def cond(cover):
        return tf.reduce_any(tf.matmul(action_set, tf.expand_dims(cover, 2)) < 1)

    cover = tf.zeros([batch_size, num_actions], tf.int32)
    cover = tf.while_loop(cond, body, [cover])

    return cover

def ccavg_action_selector(observations, q_func_online, q_func_target, **kwargs):
    """ CCAV voting aggregator as action selector
    Args:
        observations: observations tensor batch.
        q_func_online: current q_function (could be perturbed).
	q_func_target: target q_function.
	kwargs: remaining arguments if passed(ignored).

    Returns:
        batch of action choices.

    """
    batch_size = tf.shape(observations)[0]
    q_values_online = q_func_online(observations)
    q_values_target = q_func_target(observations)
    num_agents = tf.shape(q_values_online)[1]
    num_actions = tf.shape(q_values_online)[2]

    policy_action = tf.argmax(q_values_online, axis=2)
    cover = tf.reduce_max(tf.one_hot(policy_action, num_actions), 1)

    deterministic_actions = tf.argmax(tf.cast(cover, dtype=tf.float32) * tf.random_uniform((batch_size, num_actions)), axis=1)
    return deterministic_actions


def ccav_action_selector(observations, q_func_online, q_func_target, **kwargs):
    """ CCAV voting aggregator as action selector
    Args:
        observations: observations tensor batch.
        q_func_online: current q_function (could be perturbed).
	q_func_target: target q_function.
	kwargs: remaining arguments if passed(ignored).

    Returns:
        batch of action choices.

    """
    batch_size = tf.shape(observations)[0]
    q_values_online = q_func_online(observations)
    q_values_target = q_func_target(observations)
    num_agents = tf.shape(q_values_online)[1]
    num_actions = tf.shape(q_values_online)[2]

    policy_action = tf.argmax(q_values_target, axis=2)
    q_threshold_values = tf.reduce_sum(q_values_online * tf.one_hot(policy_action, num_actions), 2)
    q_threshold_values = tf.expand_dims(q_threshold_values, axis=2)

    action_sets = tf.where((q_values_online - tf.tile(q_threshold_values, [1, 1, num_actions])) >= 0,
                          tf.ones([batch_size, num_agents, num_actions], tf.int32),
                          tf.zeros([batch_size, num_agents, num_actions], tf.int32))

    cover = ccav_aggregator(action_sets)
    deterministic_actions = tf.argmax(tf.cast(cover, dtype=tf.float32) * tf.random_uniform((batch_size, num_actions)), axis=1)
    return deterministic_actions


def bootstrap_action_selector(observations, q_func_online, q_func_target, agent_ph=0, **kwargs):
    """ Bootstrap action selector 
    Args:
        observations: observations tensor batch.
        q_func_online: current q_function (could be perturbed).
	q_func_target: target q_function.
	agent_ph: Agent for the action_selection
	kwargs: remaining arguments if passed(ignored).

    Returns:
        batch of action choices.

    """
    q_func_target(observations)
    deterministic_actions = tf.argmax(tf.gather(q_func_online(observations), agent_ph, axis=1), 1)
    return deterministic_actions


