from gym.envs.registration import register

register(
    'Shapes-v0',
    entry_point='slotformer.rl.envs.block_pushing:BlockPushing',
    max_episode_steps=1050,
    kwargs={'render_type': 'shapes', 'width': 8, 'height': 8}
)
register(
    'Cubes-v0',
    entry_point='slotformer.rl.envs.block_pushing:BlockPushing',
    max_episode_steps=1050,
    kwargs={'render_type': 'cubes'},
)


register(
    'Navigation5x5-v0',
    entry_point='slotformer.rl.envs.shapes:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)


register(
    'Pushing5x5-v0',
    entry_point='envs.shapes:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': False,
        'do_reward_push_only': True,
    },
)