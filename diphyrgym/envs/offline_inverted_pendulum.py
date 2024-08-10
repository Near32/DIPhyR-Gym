from typing import Dict, List, Optional

import os
import re
import gym
from gym import spaces 
from gym.utils import seeding 
import copy
import numpy as np

from diphyrgym.envs.inverted_pendulum import InvertedPendulumDIPhiREnv
from diphyrgym.utils import STR2BT, BT2STR


class OfflineInvertedPendulumDIPhiREnv(gym.Env):
    def __init__(
        self, 
        max_nbr_actions=10,
        timestep=0.0165,
        frame_skip=1,
        max_sentence_length=16384,
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        output_dir='/run/user/{uid}/DIPhiR/inverted_pendulum', 
        use_cot=False,
        **kwargs,
    ):
        super().__init__()
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.max_sentence_length = max_sentence_length 
        # Number of times the pendulum can change orientation:
        self.max_nbr_actions = max_nbr_actions
        self.use_cot = use_cot

        self.inverted_pendulum_env = InvertedPendulumDIPhiREnv(
            model_xml=model_xml, 
            output_dir=output_dir, 
            timestep=self.timestep,
            frame_skip=self.frame_skip,
            **kwargs,
        )
        
        self.observation_space = self.inverted_pendulum_env.observation_space
        self.action_space = spaces.Discrete(n=self.max_nbr_actions)
    
    def generate_cot(self, logs):
        cot_text = []
        timestamps = np.asarray(re.findall(r'Time:\s+([\d\.]+)', logs))
        pre_rc_tstamps = timestamps[self.final_rotation_change_indices]
        post_rc_tstamps = timestamps[self.final_rotation_change_indices+1]
        assert len(pre_rc_tstamps) == len(post_rc_tstamps)
        for nidx, (pre_rc_ts, post_rc_ts) in enumerate(zip(pre_rc_tstamps, post_rc_tstamps)):
            cot_text.append(f"From time={pre_rc_ts} to time={post_rc_ts}, the sign of the angular velocity of the pole over the y axis changes, meaning that there was a change of rotation direction.\n")
            if nidx == 0:
                nth_text = "This was the 1st change of rotation direction.\n"
            elif nidx == 1:
                nth_text = "This was the 2nd change of rotation direction.\n"
            elif nidx == 2:
                nth_text = "This was the 3rd change of rotation direction.\n"
            else:
                nth_text = f"This was the {nidx+1}-th change of rotation direction.\n"

            cot_text.append(nth_text)

        cot_text = ''.join(cot_text)    
        return cot_text

    def _generate_prompt_options(self, logs):
        ''' 
        WARNING: excluding 0 again 
        '''
        '''
        prompt = f"[INST]\n{logs}\n"
        prompt += f"Given the simulation trace above, answer the following question:\n"
        prompt += "Question: How many times did the pendulum change its direction of rotation?\n"
        prompt += "[/INST]\n\n"
        '''
        prompt = f"Below is the simulation trace of a cart pole/inverted pendulum system,"
        prompt += f" followed by some instructions:\n\n{logs}\n\n"
        prompt += f"Given the simulation trace for a cart pole system above, answer the following question:\n"
        prompt += "Question: How many times did the pole change its direction of rotation?\n"
        prompt += "\n\n"
        
        if self.use_cot:
            prompt += self.generate_cot(logs)

        self.prompts = [ prompt ]
        self.options = [[
            f'Answer: {x} time{"s" if x>1 else ""}.'
            for x in range(1, self.max_nbr_actions+1) # excluding 0 here
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
        '''
        WARNING: excluding option 0 
        '''
        desired_nbr_rotation_changes = self.np_random.integers(1, self.max_nbr_actions+1)
        offset_nbr_rotation_changes = self.np_random.integers(0, self.max_nbr_actions+1)
        
        self.obs, self.info = list(self.inverted_pendulum_env.reset(**kwargs))

        # Collect infos and pendulum's angular velocity:
        infos = []
        theta_dots = []
        self.nbr_rotation_changes = 0
        while self.nbr_rotation_changes < desired_nbr_rotation_changes + offset_nbr_rotation_changes:    
            a = np.zeros(self.inverted_pendulum_env.action_space.shape)
            self.obs, reward, done, truncation, info = self.inverted_pendulum_env.step(a)
            theta_dots.append(self.inverted_pendulum_env.robot.theta_dot) 
            infos.append(info)
        
            # Compute how many times did the pendulum change direction of rotation: 
            nptheta_dots = np.array(theta_dots)
            self.rotation_change_indices = np.where(nptheta_dots[:-1] * nptheta_dots[1:] < 0)[0]
            self.nbr_rotation_changes = len(self.rotation_change_indices)
        
        # Start and end indices for the slice of logs with exactly desired_nbr_rotation_changes:
        # WARNING: if start_RC_idx and end_RC_idx points to the same index, meaning that desired_nbr_rotation_changes=1,
        # then start_iidx and end_iidx can be as close as 3 timesteps.
        # Due to the high frameskip, it is possible that theta_dots[start_iidx]<0 and theta_dots[end_iidx]<0 too because
        # only the one timestep in the middle has positive theta_dots, meaning that we actually have 2 rotation changes,
        # instead of the expected one.
        # To guard against this, we cut out the slice by decreasing end_iidx by 1.

        # WARNING: still due to the high frequency, it is also possible that it occurs when desired_nbr_rotation_changes > 1.
        # Indeed, the final timesteps may highlight rotation changes with only one timestep apart. It means that when we 
        # we slice for end_iidx, it will be 1 timestep too far, because it will include one more rotation change than expected.
        # To guard against this, we cut out the slice by decreasing end_iidx by 1, fortunately it is the same solution for all cases.
        start_RC_idx = self.np_random.integers(0, offset_nbr_rotation_changes) if offset_nbr_rotation_changes > 0 else 0
        end_RC_idx = start_RC_idx + desired_nbr_rotation_changes-1 #would cause issue if desired_nbr_rotation_changes=0
        assert end_RC_idx < len(self.rotation_change_indices)
        if start_RC_idx==0 \
        or self.rotation_change_indices[start_RC_idx-1]+1 == self.rotation_change_indices[start_RC_idx]:
            start_iidx = max(self.rotation_change_indices[start_RC_idx]-1, 0)
        else:
            start_iidx = self.np_random.integers(self.rotation_change_indices[start_RC_idx-1]+1, self.rotation_change_indices[start_RC_idx])
        if end_RC_idx==len(self.rotation_change_indices)-1 \
        or self.rotation_change_indices[end_RC_idx]+1 == self.rotation_change_indices[end_RC_idx+1]:
            end_iidx = min(self.rotation_change_indices[end_RC_idx]+1, len(infos))
        else:
            end_iidx = self.np_random.integers(self.rotation_change_indices[end_RC_idx]+1, self.rotation_change_indices[end_RC_idx+1])

        # Regularisation:
        self.nbr_rotation_changes = desired_nbr_rotation_changes
        self.rotation_change_indices = [idx-start_iidx for idx in self.rotation_change_indices[start_RC_idx:end_RC_idx+1]]
        assert len(self.rotation_change_indices) == desired_nbr_rotation_changes
        self.nptheta_dots_final = np.array(theta_dots[start_iidx:end_iidx+1])
        reg_required = False
        self.final_rotation_change_indices = np.where(self.nptheta_dots_final[:-1] * self.nptheta_dots_final[1:] < 0)[0]
        if not self.final_rotation_change_indices.shape[0] == desired_nbr_rotation_changes:
            assert end_iidx-1 > start_iidx
            reg_required = True 
            end_iidx -= 1 # WARNING : cf above
            self.nptheta_dots_final = self.nptheta_dots_final[:-1]
            self.final_rotation_change_indices = self.final_rotation_change_indices[:-1]

        # Check last info contains logs:
        collated_info = {}
        collated_info['extras'] = {'logs': []}
        for info in infos[start_iidx:end_iidx+1]: 
            for log in info['logs']: 
                collated_info['extras']['logs'].append(log)
        collated_info['extras']['log'] = '\n'.join(['\n'.join(l) for l in collated_info['extras']['logs']])
        
        # Add prompt+options:
        collated_info['prompt'] = self._generate_prompt_options(collated_info['extras']['log'])
        self.info = collated_info
        self.info['predicted_answer'] = -1 
        self.info['groundtruth_answer'] = self.nbr_rotation_changes 
        self.info['rotation_change_indices'] = self.rotation_change_indices
        
        action_mask=np.zeros((1, self.action_space.n))
        self.info['action_mask'] = action_mask
        self.info['legal_actions'] = action_mask
        return tuple([self.obs.astype(np.float32), self.info])    

    def step(self, a, **kwargs):
        '''
        Returns +1 reward if the action corresponds to the number of times the pendulum rotated. 
        Else, returns -1.
        '''
        predicted_nbr_rotation_changes = a+1 # 0 being excluded, answers are at least 1
        self.info['predicted_answer'] = predicted_nbr_rotation_changes
        self.info['groundtruth_answer'] = self.nbr_rotation_changes 
        self.info['rotation_change_indices'] = self.rotation_change_indices
        self.reward = 1 if predicted_nbr_rotation_changes == self.nbr_rotation_changes else -1
        done = True

        return self.obs.astype(np.float32), self.reward, done, False, self.info

    def close(self, **kwargs):
        self.inverted_pendulum_env.close() 

