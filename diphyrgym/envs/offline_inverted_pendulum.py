from typing import Dict, List, Optional

import os
import gym
from gym import spaces 
from gym.utils import seeding 
import copy
import numpy as np

from diphyrgym.envs.inverted_pendulum import InvertedPendulumDIPhiREnv


def STR2BT(sentences, max_sentence_length=0):
    if isinstance(sentences, str):
        sentences = [sentences]
    btss = []
    for s in sentences:
        bts = np.asarray(list(bytes(s, 'utf-8')), dtype=np.uint8)
        if max_sentence_length < bts.shape[0]:  max_sentence_length = bts.shape[0]
        btss.append(bts)
    ret = np.zeros((len(btss), max_sentence_length), dtype=np.uint8)
    for bts_idx, bts, in enumerate(btss):
        ret[bts_idx, :bts.shape[0]] = bts
    return ret

def BT2STR(bt):
    sentences = []
    for idx in range(bt.shape[0]):
        sentence = "".join(map(chr,bt[idx].tolist())).replace('\x00','')
        sentences.append(sentence)
    return sentences


class OfflineInvertedPendulumDIPhiREnv(gym.Env):
    def __init__(
        self, 
        max_nbr_actions=10,
        max_nbr_timesteps=10,
        timestep=0.0165,
        frame_skip=1,
        max_sentence_length=16384,
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        output_dir='/run/user/1000/DIPhiR/inverted_pendulum', 
        **kwargs,
    ):
        super().__init__()
        self.max_nbr_timesteps = max_nbr_timesteps
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.max_sentence_length = max_sentence_length 
        # Number of times the pendulum can change orientation:
        self.max_nbr_actions = max_nbr_actions
        
        self.inverted_pendulum_env = InvertedPendulumDIPhiREnv(
            model_xml=model_xml, 
            output_dir=output_dir, 
            timestep=self.timestep,
            frame_skip=self.frame_skip,
            **kwargs,
        )
        
        self.observation_space = self.inverted_pendulum_env.observation_space
        self.action_space = spaces.Discrete(n=self.max_nbr_actions)
    
    def _generate_prompt_options(self, logs):
        prompt = f"[INST]\n{logs}\n"
        prompt += f"Given the simulation trace above, answer the following question:"
        prompt += "Question: How many times did the pendulum change its direction of rotation?\n"
        prompt += "[/INST]\n\n"
        self.prompts = [ prompt ]
        
        self.options = [[
            f'Answer: {x} time{"s" if x>1 else ""}.'
            for x in range(0, self.max_nbr_actions)
            ],
        ]
        
        opt_sentences = []
        for pidx, prompt in enumerate(self.prompts):
            opt_prompt = prompt+'[/PROMPT]'
            opt_sentence = "[OPTION]".join(self.options[pidx])
            opt_sentences.append(opt_prompt+opt_sentence)
    
        self.bt_opt_sentences = STR2BT(opt_sentences, max_sentence_length=self.max_sentence_length)
        return self.bt_opt_sentences
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.inverted_pendulum_env.seed(seed)
        return [seed]
 
    def reset(self, **kwargs):
        self.obs, self.info = list(self.inverted_pendulum_env.reset(**kwargs))

        # Collect infos and pendulum's angular velocity:
        infos = []
        theta_dots = []
        for t in range(self.max_nbr_timesteps):
            a = np.zeros(self.inverted_pendulum_env.action_space.shape)
            self.obs, reward, done, truncation, info = self.inverted_pendulum_env.step(a)
            theta_dots.append(self.inverted_pendulum_env.robot.theta_dot) 
            infos.append(info)
        
        # Compute how many times did the pendulum change direction of rotation: 
        theta_dots = np.array(theta_dots)
        self.rotation_change_indices = np.where(theta_dots[:-1] * theta_dots[1:] < 0)[0]
        self.nbr_rotation_changes = len(self.rotation_change_indices)

        # Check last info contains logs:
        collated_info = {}
        collated_info['extras'] = {'logs': []}
        for info in infos: 
            for log in info['logs']: 
                collated_info['extras']['logs'].append(log)
        collated_info['extras']['log'] = '\n'.join(['\n'.join(l) for l in collated_info['extras']['logs']])

        # Add prompt+options:
        collated_info['prompt'] = self._generate_prompt_options(collated_info['extras']['log'])
        self.info = collated_info
        self.info['predicted_answer'] = -1 
        self.info['groundtruth_answer'] = self.nbr_rotation_changes 
        
        action_mask=np.zeros((1, self.action_space.n))
        self.info['action_mask'] = action_mask
        self.info['legal_actions'] = action_mask
        return tuple([self.obs.astype(np.float32), self.info])    

    def step(self, a, **kwargs):
        '''
        Returns +1 reward if the action corresponds to the number of times the pendulum rotated. 
        Else, returns -1.
        '''
        predicted_nbr_rotation_changes = a
        self.info['predicted_answer'] = predicted_nbr_rotation_changes
        self.info['groundtruth_answer'] = self.nbr_rotation_changes 
        self.info['rotation_change_indices'] = self.rotation_change_indices
        self.reward = 1 if predicted_nbr_rotation_changes == self.nbr_rotation_changes else -1
        done = True

        return self.obs.astype(np.float32), self.reward, done, False, self.info

    def close(self, **kwargs):
        self.inverted_pendulum_env.close() 

