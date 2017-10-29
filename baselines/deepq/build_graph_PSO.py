"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import baselines.common.tf_util as U

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var.name.split("/")[-1].split(":")[0]):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def default_param_noise_filter(var):
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="target_q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None,
                               param_noise_filter_func=None):
    """Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    if param_noise_filter_func is None:
        param_noise_filter_func = default_param_noise_filter

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), name="update_param_noise_threshold")
        update_param_noise_scale_ph = tf.placeholder(tf.bool, (), name="update_param_noise_scale")
        reset_ph = tf.placeholder(tf.bool, (), name="reset")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01), trainable=False)
        param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05), trainable=False)

        # Unmodified Q.
        q_values = q_func(observations_ph.get(), num_actions, scope="target_q_func")

        # Perturbable Q used for the actual rollout.
        q_values_perturbed = q_func(observations_ph.get(), num_actions, scope="perturbed_q_func")
        # We have to wrap this code into a function due to the way tf.cond() works. See
        # https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
        # a more detailed discussion.
        def perturb_vars(original_scope, perturbed_scope):
            all_vars = U.scope_vars(U.absolute_scope_name("q_func"))
            all_perturbed_vars = U.scope_vars(U.absolute_scope_name("perturbed_q_func"))
            assert len(all_vars) == len(all_perturbed_vars)
            perturb_ops = []
            for var, perturbed_var in zip(all_vars, all_perturbed_vars):
                if param_noise_filter_func(perturbed_var):
                    # Perturb this variable.
                    op = tf.assign(perturbed_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=param_noise_scale))
                else:
                    # Do not perturb, just assign.
                    op = tf.assign(perturbed_var, var)
                perturb_ops.append(op)
            assert len(perturb_ops) == len(all_vars)
            return tf.group(*perturb_ops)

        # Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
        # of the network and measures the effect of that perturbation in action space. If the perturbation
        # is too big, reduce scale of perturbation, otherwise increase.
        q_values_adaptive = q_func(observations_ph.get(), num_actions, scope="adaptive_q_func")
        perturb_for_adaption = perturb_vars(original_scope="target_q_func", perturbed_scope="adaptive_q_func")
        kl = tf.reduce_sum(tf.nn.softmax(q_values) * (tf.log(tf.nn.softmax(q_values)) - tf.log(tf.nn.softmax(q_values_adaptive))), axis=-1)
        mean_kl = tf.reduce_mean(kl)
        def update_scale():
            with tf.control_dependencies([perturb_for_adaption]):
                update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                    lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                    lambda: param_noise_scale.assign(param_noise_scale / 1.01),
                )
            return update_scale_expr

        # Functionality to update the threshold for parameter space noise.
        update_param_noise_threshold_expr = param_noise_threshold.assign(tf.cond(update_param_noise_threshold_ph >= 0,
            lambda: update_param_noise_threshold_ph, lambda: param_noise_threshold))

        # Put everything together.
        deterministic_actions = tf.argmax(q_values_perturbed, axis=1)
        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        updates = [
            update_eps_expr,
            tf.cond(reset_ph, lambda: perturb_vars(original_scope="target_q_func", perturbed_scope="perturbed_q_func"), lambda: tf.group(*[])),
            tf.cond(update_param_noise_scale_ph, lambda: update_scale(), lambda: tf.Variable(0., trainable=False)),
            update_param_noise_threshold_expr,
        ]
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph, update_param_noise_scale_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True, reset_ph: False, update_param_noise_threshold_ph: False, update_param_noise_scale_ph: False},
                         updates=updates)
        return act


def build_train_PSO(make_obs_ph, q_func, num_actions, optimizer, num_clones, inertia=float(1), cognitive=float(1),
                    social=float(1), grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None,
                    param_noise=False, param_noise_filter_func=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """

    with tf.variable_scope(scope, reuse=reuse):

        act_f_clones = []
        train_clones = []
        update_target_clones = []
        q_value_clones = []

        clone_returns_ph = tf.placeholder(tf.float32, [num_clones], name="clone_returns")

        group_best_return = tf.get_variable("group_best_return", (), initializer=tf.constant_initializer(float('-inf')))
        tf.summary.scalar("group_best_return", group_best_return)
        obs_dummy_input = U.ensure_tf_input(make_obs_ph("obs_dummy"))
        group_best_q_func = q_func(obs_dummy_input.get(), num_actions, scope="group_best_q_func")
        group_best_q_func_vars = U.scope_vars(U.absolute_scope_name("group_best_q_func"))

        with tf.name_scope("Group_best_network"):
            for q_var in sorted(group_best_q_func_vars, key=lambda v: v.name):
                variable_summaries(q_var)

        update_clone_best_expr = []
        update_group_best_expr = []
        update_clone_velocity_expr = []
        update_clone_q_expr = []
        clone_best_returns = []

        inertia = tf.fill([], inertia, "inertia")
        cognitive = tf.fill([], cognitive, "cognitive")
        social = tf.fill([], social, "social")

        for clone in range(num_clones):
            clone_scope = "clone{}".format(clone)
            if param_noise:
                act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=clone_scope, reuse=reuse,
                                                   param_noise_filter_func=param_noise_filter_func)
            else:
                act_f = build_act(make_obs_ph, q_func, num_actions, scope=clone_scope, reuse=reuse)

            with tf.variable_scope(clone_scope, reuse=reuse):

                # set up placeholders
                obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
                act_t_ph = tf.placeholder(tf.int32, [None], name="action")
                rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
                obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
                done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
                importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

                # q network evaluation
                q_t = q_func(obs_t_input.get(), num_actions, scope="q_func")  # reuse parameters from act
                q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

                with tf.name_scope("Live_Network"):
                    for q_var in sorted(q_func_vars, key=lambda v: v.name):
                        variable_summaries(q_var)

                # target q network evalution
                q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func", reuse=True)
                target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

                with tf.name_scope("Target_Network"):
                    for target_q_var in sorted(target_q_func_vars, key=lambda v: v.name):
                        variable_summaries(target_q_var)

                # q scores for actions which we know were selected in the given state.
                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

                # compute estimate of best possible value starting from state at t + 1
                if double_q:
                    q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
                    q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
                    q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
                else:
                    q_tp1_best = tf.reduce_max(q_tp1, 1)
                q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

                # compute RHS of bellman equation
                q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

                # compute the error (potentially clipped)
                td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
                errors = U.huber_loss(td_error)
                weighted_error = tf.reduce_mean(importance_weights_ph * errors)

                # with tf.name_scope("TD_error"):
                #     tf.summary.scalar(td_error.name, td_error)

                # compute optimization op (potentially with gradient clipping)
                if grad_norm_clipping is not None:
                    optimize_expr = U.minimize_and_clip(optimizer,
                                                        weighted_error,
                                                        var_list=q_func_vars,
                                                        clip_val=grad_norm_clipping)
                else:
                    optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

                # update_target_fn will be called periodically to copy Q network to target Q network
                update_target_expr = []
                for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                           sorted(target_q_func_vars, key=lambda v: v.name)):
                    update_target_expr.append(var_target.assign(var))
                update_target_expr = tf.group(*update_target_expr)

                # merged = tf.summary.merge_all()

                # Create callable functions
                train = U.function(
                    inputs=[
                        obs_t_input,
                        act_t_ph,
                        rew_t_ph,
                        obs_tp1_input,
                        done_mask_ph,
                        importance_weights_ph
                    ],
                    outputs=[td_error],
                    updates=[optimize_expr]
                )
                update_target = U.function([], [], updates=[update_target_expr])

                q_values = U.function([obs_t_input], q_t)

                # Velocity variable and best q_func variable of clones
                velocity_q_func = q_func(obs_t_input.get(), num_actions, scope="velocity_q_func")
                velocity_q_func_vars = U.scope_vars(U.absolute_scope_name("velocity_q_func"))

                with tf.name_scope("Velocity_network"):
                    for q_var in sorted(velocity_q_func_vars, key=lambda v: v.name):
                        variable_summaries(q_var)

                clone_best_q_func = q_func(obs_t_input.get(), num_actions, scope="clone_best_q_func")
                clone_best_q_func_vars = U.scope_vars(U.absolute_scope_name("clone_best_q_func"))

                with tf.name_scope("Clone_best_network"):
                    for q_var in sorted(clone_best_q_func_vars, key=lambda v: v.name):
                        variable_summaries(q_var)

                clone_best_return = tf.get_variable("clone_best_return", (),
                                                    initializer=tf.constant_initializer(float('-inf')))
                tf.summary.scalar("clone_best_return", clone_best_return)
                clone_best_returns.append(clone_best_return)

                #PSO that updates the target network
                for target_q_var, clone_best_q_var, group_best_q_var in zip(
                        sorted(target_q_func_vars, key=lambda v: v.name),
                        sorted(clone_best_q_func_vars, key=lambda v: v.name),
                        sorted(group_best_q_func_vars, key=lambda v: v.name)):
                    update_clone_best_expr.append(
                        clone_best_q_var.assign(tf.cond(clone_returns_ph[clone] > clone_best_return,
                                                        lambda: target_q_var,
                                                        lambda: clone_best_q_var)))
                    update_group_best_expr.append(
                        group_best_q_var.assign(tf.cond(clone_returns_ph[clone] > group_best_return,
                                                        lambda: target_q_var,
                                                        lambda: group_best_q_var)))
                update_clone_best_expr.append(clone_best_return.assign(
                    tf.cond(clone_returns_ph[clone] > clone_best_return,
                            lambda: clone_returns_ph[clone],
                            lambda: clone_best_return)))
                update_group_best_expr.append(group_best_return.assign(
                    tf.cond(clone_returns_ph[clone] > group_best_return,
                            lambda: clone_returns_ph[clone],
                            lambda: group_best_return)))

                for velocity_var, clone_best_q_var, group_best_q_var, target_q_var \
                        in zip(sorted(velocity_q_func_vars, key=lambda v: v.name),
                               sorted(clone_best_q_func_vars, key=lambda v: v.name),
                               sorted(group_best_q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
                    cognitive_random = tf.random_uniform([], 0, cognitive)
                    social_random = tf.random_uniform([], 0, social)
                    clone_best_difference = tf.subtract(clone_best_q_var, target_q_var)
                    group_best_difference = tf.subtract(group_best_q_var, target_q_var)
                    new_velocity_var = tf.add_n([tf.multiply(inertia, velocity_var),
                                                 tf.multiply(cognitive_random, clone_best_difference),
                                                 tf.multiply(social_random, group_best_difference)])
                    update_clone_velocity_expr.append(velocity_var.assign(new_velocity_var))

                for target_q_var, velocity_var in zip(sorted(target_q_func_vars, key=lambda v: v.name),
                                                      sorted(velocity_q_func_vars, key=lambda v: v.name)):
                    update_clone_q_expr.append(target_q_var.assign(tf.add(target_q_var, velocity_var)))

                #PSO that updates the current network
                # for q_var, clone_best_q_var, group_best_q_var in zip(
                #         sorted(q_func_vars, key=lambda v: v.name),
                #         sorted(clone_best_q_func_vars, key=lambda v: v.name),
                #         sorted(group_best_q_func_vars, key=lambda v: v.name)):
                #     if q_var.name.endswith('weights:0'):
                #         update_clone_best_expr.append(
                #             clone_best_q_var.assign(tf.cond(clone_returns_ph[clone] > clone_best_return,
                #                                             lambda: q_var,
                #                                             lambda: clone_best_q_var)))
                #         update_group_best_expr.append(
                #             group_best_q_var.assign(tf.cond(clone_returns_ph[clone] > group_best_return,
                #                                             lambda: q_var,
                #                                             lambda: group_best_q_var)))
                # update_clone_best_expr.append(clone_best_return.assign(
                #     tf.cond(clone_returns_ph[clone] > clone_best_return,
                #             lambda: clone_returns_ph[clone],
                #             lambda: clone_best_return)))
                # update_group_best_expr.append(group_best_return.assign(
                #     tf.cond(clone_returns_ph[clone] > group_best_return,
                #             lambda: clone_returns_ph[clone],
                #             lambda: group_best_return)))
                #
                # for velocity_var, clone_best_q_var, group_best_q_var, q_var \
                #         in zip(sorted(velocity_q_func_vars, key=lambda v: v.name),
                #                sorted(clone_best_q_func_vars, key=lambda v: v.name),
                #                sorted(group_best_q_func_vars, key=lambda v: v.name),
                #                sorted(q_func_vars, key=lambda v: v.name)):
                #     if velocity_var.name.endswith('weights:0'):
                #         update_clone_velocity_expr.append(velocity_var.assign(
                #             tf.add_n([tf.multiply(inertia, velocity_var),
                #                       tf.multiply(cognitive, tf.multiply(tf.random_uniform([], 0, 1),
                #                                                          tf.subtract(clone_best_q_var, q_var))),
                #                       tf.multiply(social, tf.multiply(tf.random_uniform([], 0, 1),
                #                                                       tf.subtract(group_best_q_var, q_var)))])))
                #
                # for q_var, velocity_var in zip(sorted(q_func_vars, key=lambda v: v.name),
                #                                       sorted(velocity_q_func_vars, key=lambda v: v.name)):
                #     update_clone_q_expr.append(q_var.assign(tf.add(q_var, velocity_var)))



                act_f_clones.append(act_f)
                train_clones.append(train)
                update_target_clones.append(update_target)
                q_value_clones.append(q_values)

        update_clone_best_expr = tf.group(*update_clone_best_expr)
        update_group_best_expr = tf.group(*update_group_best_expr)
        update_clone_velocity_expr = tf.group(*update_clone_velocity_expr)
        update_clone_q_expr = tf.group(*update_clone_q_expr)

        merged = tf.summary.merge_all()

        pso_update = U.function(
            inputs=[clone_returns_ph],
            outputs=[merged,
                     group_best_return,
                     clone_best_returns,
                     group_best_q_func_vars],
            updates=[update_clone_best_expr,
                     update_group_best_expr,
                     update_clone_velocity_expr,
                     update_clone_q_expr]
        )

        return act_f_clones, train_clones, update_target_clones, q_value_clones, pso_update
