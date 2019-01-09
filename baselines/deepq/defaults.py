def atari():
    return dict(
        network='conv_only',
        lr=1e-4,
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.0001,
        train_freq=4,
        learning_starts=50000,
        target_network_update_freq=5000,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=False,
        num_agents=10,
        print_freq=1
    )

def retro():
    return atari()

