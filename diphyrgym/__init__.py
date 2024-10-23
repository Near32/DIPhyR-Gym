'''
Licence.
'''
from diphyrgym.envs import *

import gym
from gym.envs.registration import register

for env_k in list(gym.envs.registration.registry.keys()):
    if 'DIPhyR' in env_k:
        del gym.envs.registration.registry[env_k]

'''
for n in range(2,10):
    register(
        id=f'DIPhiR-{n}ObjectClasses-v0',
        entry_point='src.env:DIPhiREnv',
        kwargs={'n' : n},
    )
'''

## offline pendula
register(
	id='OfflineInvertedPendulumDIPhyREnv-v0',
	entry_point='diphyrgym.envs.offline_inverted_pendulum:OfflineInvertedPendulumDIPhyREnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
  order_enforce=False,
	)

## pendula
register(
	id='InvertedPendulumDIPhyREnv-v0',
	entry_point='diphyrgym.envs.inverted_pendulum:InvertedPendulumDIPhyREnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
  order_enforce=False,
	)

register(
	id='InvertedDoublePendulumDIPhyREnv-v0',
	entry_point='diphyrgym.envs.inverted_double_pendulum:InvertedDoublePendulumDIPhyREnv',
	max_episode_steps=1000,
	reward_threshold=9100.0,
  order_enforce=False,
	)

## locomotors 
register(
	id='HopperDIPhyREnv-v0',
	entry_point='diphyrgym.envs.hopper:HopperDIPhyREnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
  order_enforce=False,
	)


