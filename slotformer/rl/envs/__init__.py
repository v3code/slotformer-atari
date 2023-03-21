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
