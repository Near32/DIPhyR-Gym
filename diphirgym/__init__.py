from .env import *
from .utils import *

import gym
from gym.envs.registration import register

for env_k in gym.envs.registration.registry.env_specs.keys():
    if 'DIPhiR' in env_k:
        del gym.envs.registration.registry.env_specs[env_k]

for n in range(2,10):
    register(
        id=f'DIPhiR-{n}ObjectClasses-v0',
        entry_point='src.env:DIPhiREnv',
        kwargs={'n' : n},
    )

