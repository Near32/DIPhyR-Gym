'''
Licence.
'''
from typing import Dict, List, Optional

import os
import time
import re
import gym
from gym import spaces 
from gym.utils import seeding 
import copy
import numpy as np

from diphyrgym.envs.inverted_pendulum import InvertedPendulumDIPhyREnv
from diphyrgym.utils import STR2BT, BT2STR


class OfflineInvertedPendulumDIPhyREnv(gym.Env):
    def __init__(
        self, 
        max_nbr_actions={
            'pole': 10, 
            'cart': 40,
        },
        timestep=0.0165,
        frame_skip=1,
        max_sentence_length=16384,
        model_xml=os.path.join(os.path.dirname(__file__), "../xmls/inverted_pendulum.xml"), 
        output_dir='/run/user/{uid}/DIPhyR/inverted_pendulum', 
        use_cot=False,
        notrace=False,
        prompt_types=[
            'pole_min_angular',
            'pole_max_angular',
            'cart_min_linear',
            'cart_max_linear',
            'pole_longest_time',
            'pole_shortest_time',
            'cart_longest_time',
            'cart_shortest_time',
            'pole',
            'cart',
        ],
        **kwargs,
    ):
        '''
        max_nbr_actions: Dict[str,int] of maximums number of actions, which defines the maximum 
            number of options for each question defined by each prompt types, e.g. for the 'pole' 
            entry it corresponds to the maximum number of times the pendulum can change its direction
            of rotation. For the 'cart' entry, it corresponds to the maximum number of times the cart
            may change its direction of movement. A good rule-of-thumb is to set it to 4 times the pole's.
                - {pole/cart}_{min/max}_{angular/linear} : corresponds to the number of timesteps 
                visible in the simulation trace : the default value is set to use the pole's 
                max number of rotation direction multiplied by 8*24 and divided by :param fame_skip:.
                - {pole/cart}_{shortest/longest}_time : corresponds to the {shortest/longest} time 
                period in the simulation trace where the {pole/cart} movement direction remained 
                constant : the default value is set to use the pole's max number of rotation 
                direction multiplied by 24 and divided by :param fame_skip:.
        timestep: simulation timestep.
        frame_skip: number of frameskips, in order to reduce the size of the simulation trace.
        max_sentence_length: maximum length of the resulting prompt, UTF-8 encoded characters.
        model_xml: path to the model xml, which should contain some extra variables to provide
            the environment with boundary conditions, like initial angular and linear velocities.
        output_dir: path to the directory where the xmls and cot files will be saved. It is best
            to use a tmpfs path, so that the complexity of the environment can be reduced.
        use_cot: whether to include an accurate rule-based-generated Chain-of-Thought reasoning
            into the prompt.
        notrace: whether to not include the simulation trace in the prompt. It is necessary to
            use CoT if True.
        prompt_types: List[str] of the prompt types to use. Currently supports:
            - 'pole': How many times did the pole change its direction of rotation? 
            - 'cart': How many times did the cart change its direction of translation?
            - '{pole/cart}_{min/max}_{angular/linear}': What is the timestamp id where the 
            absolute {angular/linear} velocity of the {pole/cart} was at its {maximum/minimum} ?
        '''
        if kwargs.get('minimal_logs', False):
            for pt in prompt_types:
                if 'cart' in pt:
                    raise NotImplementedError(
                        "Using minimal logs with the cart prompt types is not implemented yet." \
                        "\nTODO: the base_env from pybullet needs to return minimal_logs that are" \
                        "conditioned on the prompt_types..." \
                        "\nCf. reset method where it collates the logs from the base env..."
                    )

        super().__init__()
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.max_sentence_length = max_sentence_length 
        self.prompt_types = prompt_types #['pole','cart']
        # Number of times the pendulum can change orientation:
        self.max_nbr_actions = max_nbr_actions
        if not isinstance(self.max_nbr_actions, dict):
            assert isinstance(self.max_nbr_actions, int)
            self.max_nbr_actions = {}
            if 'pole_min_angular' in self.prompt_types:
                self.max_nbr_actions['pole_min_angular'] = int(24/frame_skip)*8*self.max_nbr_actions,
            if 'pole_max_angular' in self.prompt_types:
                self.max_nbr_actions['pole_max_angular'] = int(24/frame_skip)*8*self.max_nbr_actions,
            if 'cart_min_linear' in self.prompt_types:
                self.max_nbr_actions['cart_min_linear'] = int(24/frame_skip)*8*self.max_nbr_actions,
            if 'cart_max_linear' in self.prompt_types:
                self.max_nbr_actions['cart_max_linear'] = int(24/frame_skip)*8*self.max_nbr_actions,
            if 'pole_longest_time' in self.prompt_types:
                self.max_nbr_actions['pole_longest_time'] = int(24/frame_skip)*self.max_nbr_actions,
            if 'pole_shortest_time' in self.prompt_types:
                self.max_nbr_actions['pole_shortest_time'] = int(24/frame_skip)*self.max_nbr_actions,
            if 'cart_shortest_time' in self.prompt_types:
                self.max_nbr_actions['cart_shortest_time'] = int(24/frame_skip)*self.max_nbr_actions,
            if 'cart_longest_time' in self.prompt_types:
                self.max_nbr_actions['cart_longest_time'] = int(24/frame_skip)*self.max_nbr_actions,
            if 'pole' in self.prompt_types:
                self.max_nbr_actions['pole'] = self.max_nbr_actions, 
            if 'cart' in self.prompt_types:
                self.max_nbr_actions['cart'] = 4*self.max_nbr_actions,
            
        self.use_cot = use_cot
        self.notrace = notrace
        
        self.inverted_pendulum_env = InvertedPendulumDIPhyREnv(
            model_xml=model_xml, 
            output_dir=output_dir, 
            timestep=self.timestep,
            frame_skip=self.frame_skip,
            **kwargs,
        )
        
        self.observation_space = self.inverted_pendulum_env.observation_space
        self.action_space = spaces.Discrete(n=max(self.max_nbr_actions.values()))
        
        self.prompts = {}
 
    def generate_cot(
        self, 
        logs,
        prompt_type='pole',
    ):
        '''
        Generates the Chain-of-Thought reasoning.
        :param prompt_type: str within 
            - 'pole' ;
            - 'cart' ;
            - '{pole/cart}_{min/max}_{angular/linear}' ;
            - '{pole/cart}_{shortest/longest}_time' ;
        '''
        if '_time' in prompt_type \
        and ('pole_' in prompt_type \
        or 'cart_' in prompt_type):
            rgb, shortlong, time = prompt_type.split('_')
            assert time=='time'
            return self._generate_cot_shortlong_time(
                logs=logs,
                rgb=rgb,
                shortlong=shortlong,
            )
        elif 'pole_' in prompt_type \
        or 'cart_' in prompt_type:
            rgb, minmax, anglinear = prompt_type.split('_')
            return self._generate_cot_minmax_vel(
                logs=logs,
                rgb=rgb,
                minmax=minmax,
                anglinear=anglinear,
            )
        elif 'pole'==prompt_type:
            return self._generate_cot_pole(logs)
        elif 'cart'==prompt_type:
            return self._generate_cot_cart(logs)
        else:
            raise NotImplementedError

    def _generate_cot_shortlong_time(
        self, 
        logs,
        rgb='pole',
        shortlong='shortest',
    ):
        '''
        Generates the Chain-of-Thought reasoning for the {pole/cart}_{short/long}_time Qs.
        '''
        if rgb=='pole': 
            change_indices = copy.deepcopy(self.final_rotation_change_indices)
        if rgb=='cart': 
            change_indices = copy.deepcopy(self.final_sliding_change_indices)
        if shortlong=='shortest': 
            opt_fn = np.min
            argopt_fn = np.argmin
        elif shortlong=='longest':
            opt_fn = np.max
            argopt_fn = np.argmax
        else:
            raise NotImplementedError

        timestamps = np.asarray([val for val in re.findall(r'Time:\s+([\d\.]+)', logs)])
        # Change indices points at timesteps just before change so the end of any phase is the current timestep, but the start of any phase is the next timestamp:
        start_tstamps = timestamps[change_indices+1]
        if timestamps[0] not in start_tstamps:  
            # regularisation to make sure that the phases starts at the first timestamp:
            start_tstamps = np.array([timestamps[0]]+start_tstamps.tolist())

        # What happens then if the last timestamp is part of the start_tstamps?
        # It can occur if there is a change occuring between the penultimate and the last timestamp.
        # But we have no means to know how long does the phase last, so we remove it:
        if timestamps[-1] in start_tstamps:
            # we need to remove it, because we have no idea how long it will last:
            start_tstamps = start_tstamps[:-1]

        end_tstamps = timestamps[change_indices]
        # WARNING: what happens if the first timestamp is pointed as a change indices:
        # Then we need to remove it from the end list:
        if timestamps[0] in end_tstamps:
            end_tstamps = end_tstamps[1:]
        # Because we care about phases, the last timestamps must be included as an end of phase:
        # if an only if we do not have enough phases:
        if timestamps[-1] not in end_tstamps \
        and not(len(start_tstamps)==len(end_tstamps)):
            end_tstamps = np.array(end_tstamps.tolist()+[timestamps[-1]])
        
        assert len(start_tstamps) == len(end_tstamps)
        timeperiods = [ 
            np.round(float(ets)-float(sts), decimals=3)
            for sts, ets in zip(start_tstamps, end_tstamps)
        ]

        opt_time = opt_fn(timeperiods)
        #argopt_time = argopt_fn(timeperiods)+1 # to start counting from 1

        cot_text = []
        for nidx, (sts, ets) in enumerate(zip(start_tstamps, end_tstamps)):
            diff = timeperiods[nidx]
            cot_text.append(f"From time={sts} to time={ets}, the sign of the angular velocity of the pole over the y axis remained the same, meaning that the rotation direction remained the same for {diff} secs.\n")
        
        nth_text = f"\nThus, the {shortlong} time period where the movement of the {rgb} remained the same is {opt_time} secs.\n"
        cot_text.append(nth_text)
        cot_text = ''.join(cot_text)    
        return cot_text

    def _generate_cot_minmax_vel(
        self, 
        logs,
        rgb='pole',
        minmax='min',
        anglinear='angular',
    ):
        '''
        Generates the Chain-of-Thought reasoning for the {pole/cart}_{min/max}_{angular/linear} Qs.
        WARNING: timestamps are numbered from 1, therefore excluding 0 from possible answers.
        '''
        if rgb=='pole': 
            assert anglinear=='angular'
            velocities = copy.deepcopy(self.nptheta_dots_final)
        if rgb=='cart': 
            assert anglinear=='linear'
            velocities = copy.deepcopy(self.npx_dots_final)
        if minmax=='min': 
            opt_fn = np.min
            argopt_fn = np.argmin
        elif minmax=='max':
            opt_fn = np.max
            argopt_fn = np.argmax
        else:
            raise NotImplementedError

        timestamps = np.asarray(re.findall(r'Time:\s+([\d\.]+)', logs))
        opt_vel = opt_fn(velocities)
        argopt_vel = argopt_fn(velocities)+1 # to start counting from 1

        cot_text = []
        cot_text.append(
            f"Below is the list of timestamps and the relevant {anglinear} velocities of the {rgb}:",
        )
        for tidx, (ts, vel) in enumerate(zip(timestamps, velocities)):
            nth_text = f"\n- At timestamp {tidx+1} (time={ts}) : the {rgb}'s absolute {anglinear} velocity was {np.abs(vel)}."
            cot_text.append(nth_text)

        nth_text = f"\n\nThus, the {minmax} absolute {anglinear} velocity of the {rgb} was achieved at timestamp {argopt_vel}.\n"
        cot_text.append(nth_text)
        cot_text = ''.join(cot_text)
        return cot_text

    def _generate_cot_cart(self, logs):
        '''
        Generates the Chain-of-Thought reasoning for the cart.
        '''
        cot_text = []
        timestamps = np.asarray(re.findall(r'Time:\s+([\d\.]+)', logs))
        pre_rc_tstamps = timestamps[self.final_sliding_change_indices]
        post_rc_tstamps = timestamps[self.final_sliding_change_indices+1]
        assert len(pre_rc_tstamps) == len(post_rc_tstamps)
        for nidx, (pre_rc_ts, post_rc_ts) in enumerate(zip(pre_rc_tstamps, post_rc_tstamps)):
            cot_text.append(f"From time={pre_rc_ts} to time={post_rc_ts}, the sign of the linear velocity of the cart over the x axis changes, meaning that there was a change of translation direction.\n")
            if nidx == 0:
                nth_text = "This was the 1st change of translation direction.\n"
            elif nidx == 1:
                nth_text = "This was the 2nd change of translation direction.\n"
            elif nidx == 2:
                nth_text = "This was the 3rd change of translation direction.\n"
            else:
                nth_text = f"This was the {nidx+1}-th change of translation direction.\n"

            cot_text.append(nth_text)

        if len(cot_text)==0:
            # There was no change in cart translation direction:
            pre_rc_ts = timestamps[0]; post_rc_ts = timestamps[-1]
            cot_text = f"From time={pre_rc_ts} to time={post_rc_ts}, the sign of the linear velocity of the cart over the x axis stayed the same, meaning that there was no change in the direction of the cart's translation.\n"
        
        cot_text = ''.join(cot_text)
        return cot_text

    def _generate_cot_pole(self, logs):
        '''
        Generates the Chain-of-Thought reasoning for the pole.
        '''
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

    def _generate_prompt_options(
        self, 
        logs,
        prompt_type,
    ):
        ''' 
        Generates the prompt and options.
        Reset the timestamps in the log so that it starts at t=0.
        :param prompt_type: str among:
            - 'pole' ;
            - 'cart' ;
            - '{pole/cart}_{min/max}_{angular/linear}' ;
            - '{pole/cart}_{shortest/longest}_time' ;
        WARNING: excluding 0 again from answer choices in 'pole' type. 

        :return: a list of strings, each string is a prompt option, 
        along with the groundtruth answer.
        '''
        
        timestamps = re.findall(r'Time:\s+([\d\.]+)', logs)
        t0 = float(timestamps[0])
        modified_timestamps = [float(t)-t0 for t in timestamps]
        for original, modified in zip(timestamps, modified_timestamps):
            logs = re.sub(f"Time:\s+{original}", f"Time: {modified:.3f}", logs, 1)
        
        if '_time' in prompt_type \
        and ('pole_' in prompt_type \
        or 'cart_' in prompt_type):
            rgb, shortlong, time = prompt_type.split('_')
            assert time=='time'
            return self._generate_prompt_options_shortlong_time(
                logs=logs,
                rgb=rgb,
                shortlong=shortlong,
            )
        elif 'pole_' in prompt_type \
        or 'cart_' in prompt_type:
            rgb, minmax, anglinear = prompt_type.split('_')
            return self._generate_prompt_options_minmax_vel(
                logs=logs,
                rgb=rgb,
                minmax=minmax,
                anglinear=anglinear,
            )
        elif 'pole'== prompt_type:
            return self._generate_prompt_options_pole(logs)
        elif 'cart'==prompt_type:
            return self._generate_prompt_options_cart(logs)
        else:
            raise NotImplementedError

    def _generate_prompt_options_shortlong_time(
        self, 
        logs,
        rgb='pole',
        shortlong='shortest',
    ):
        '''
        Generates the prompt and options for the '{pole/cart}_{shortest/longest}_time' QA type.
        WARNING: timestamps are numbered from 1, therefore excluding 0 from possible answers.
        
        :return: a list of strings, each string is a prompt option, along with the groundtruth answer.
        '''
        if rgb=='pole': 
            change_indices = copy.deepcopy(self.final_rotation_change_indices)
        if rgb=='cart': 
            change_indices = copy.deepcopy(self.final_sliding_change_indices)
        if shortlong=='shortest': 
            opt_fn = np.min
            argopt_fn = np.argmin
        elif shortlong=='longest':
            opt_fn = np.max
            argopt_fn = np.argmax
        else:
            raise NotImplementedError

        timestamps = np.asarray([val for val in re.findall(r'Time:\s+([\d\.]+)', logs)])
        # Change indices points at timesteps just before change so the end of any phase is the current timestep, but the start of any phase is the next timestamp:
        start_tstamps = timestamps[change_indices+1]
        if timestamps[0] not in start_tstamps:  
            # regularisation to make sure that the phases starts at the first timestamp:
            start_tstamps = np.array([timestamps[0]]+start_tstamps.tolist())

        # What happens then if the last timestamp is part of the start_tstamps?
        # It can occur if there is a change occuring between the penultimate and the last timestamp.
        # But we have no means to know how long does the phase last, so we remove it:
        if timestamps[-1] in start_tstamps:
            # we need to remove it, because we have no idea how long it will last:
            start_tstamps = start_tstamps[:-1]

        end_tstamps = timestamps[change_indices]
        # WARNING: what happens if the first timestamp is pointed as a change indices:
        # Then we need to remove it from the end list:
        if timestamps[0] in end_tstamps:
            end_tstamps = end_tstamps[1:]
        # Because we care about phases, the last timestamps must be included as an end of phase:
        # if an only if we do not have enough phases:
        if timestamps[-1] not in end_tstamps \
        and not(len(start_tstamps)==len(end_tstamps)):
            end_tstamps = np.array(end_tstamps.tolist()+[timestamps[-1]])
        
        assert len(start_tstamps) == len(end_tstamps)
        timeperiods = [ 
            np.round(float(ets)-float(sts), decimals=3)
            for sts, ets in zip(start_tstamps, end_tstamps)
        ]

        '''
        timestamps = np.asarray([float(val) for val in re.findall(r'Time:\s+([\d\.]+)', logs)])
        # Change indices point at the timestamp indices just before the change occurs.
        # So we need to take the timestampd just after that:
        start_change_indices = (change_indices+1)
        # But it is possible that the last timestamp show a change, i.e. one change index points
        # at the penultimate timestamp :
        # We cannot estimate how long the resulting time period lasted, so we drop it from
        # evaluation:
        start_change_indices = start_change_indices[start_change_indices<len(timestamps)]
        start_tstamps = timestamps[start_change_indices]
        if timestamps[0] not in start_tstamps:  
            # regularisation to make sure that the period starts at the first timestamp:
            start_tstamps = np.array([timestamps[0]]+start_tstamps.tolist())
        else:
            # this should not occur ? I can't see why would the first timestamp 
            # be considered a rotation change?
            # It can occur because the change indices point at timesteps just before the change occurs.
            # So, there is no problem, it is a possiblity:
            # We need to remove it from the end timestamps though:
            change_indices = change_indices[1:]
        end_tstamps = timestamps[change_indices]
        if timestamps[-1] not in end_tstamps:   
            end_tstamps = np.array(end_tstamps.tolist()+[timestamps[-1]])
        else:
            import ipdb; ipdb.set_trace()
            # Similarly, it is possible that the last timestamp is considered a change index:
            # In that case, we cannot compute how long did the resulting period last, so we
            # do not consider it:
            # Meaning that we remove the value from the end_tstamps:
            end_tstamps = end_tstamps[:-1]
        assert len(start_tstamps) == len(end_tstamps)
        timeperiods = [ 
            np.round(ets-sts, decimals=3)
            for sts, ets in zip(start_tstamps, end_tstamps)
        ]
        '''

        # Generate possible answers:
        opt_time = opt_fn(timeperiods)
        possible_answers = list(set(timeperiods))
        # we exclude the gt_answer in order to introduce it back randomly later:
        possible_answers = list(set(timeperiods).difference([opt_time]))

        # Is there enough candidates:
        while len(possible_answers) < self.max_nbr_actions[f"{rgb}_{shortlong}_time"]:
            new_option = (self.timestep*self.frame_skip)*self.np_random.integers(1,2*(self.max_nbr_actions['pole']+1))
            possible_answers = list(set(possible_answers+[np.round(new_option, decimals=3)]))
        # Is there too many?
        while len(possible_answers) > self.max_nbr_actions[f"{rgb}_{shortlong}_time"]:
            del possible_answers[0]
        assert len(possible_answers) == self.max_nbr_actions[f"{rgb}_{shortlong}_time"]
        
        # Generate GT answer:
        # Let us place the GT answer randomly:
        gt_answer = self.np_random.integers(0,len(possible_answers))
        possible_answers[gt_answer] = opt_time
        ##

        if self.notrace:
            assert self.use_cot
            prompt = f"Below is some information about the simulation of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:\n"
            prompt += self.generate_cot(logs, prompt_type=f"{rgb}_{shortlong}_time")
            prompt += f"You are an expert in the matter. Given the information above, answer the following question to the best of your abilities.\n"
            prompt += f"Question: What is the {shortlong} time period (in seconds) where the movement of the {rgb} remained in the same direction?\n"
            #prompt += "Question: At which timestamp was the {minmax} {anglinear} velocity of the {rgb} reached?\n"
            prompt += "\n"
        else:
            prompt = f"Below is the simulation trace of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:"
            prompt += f"\n\n{logs}\n\n"
            prompt += f"You are an expert in the matter. Given the simulation trace for a cart pole system above, answer the following question to the best of your abilities:\n"
            prompt += f"Question: What is the {shortlong} time period (in seconds) where the movement of the {rgb} remained in the same direction, without considering interpolations between timestamps?\n"
            prompt += "\n"
        
            if self.use_cot:
                prompt += self.generate_cot(logs, prompt_type=f"{rgb}_{shortlong}_time")

        prompts = [ prompt ]
        options = [[
            f'Answer: {x} seconds.'
            for x in possible_answers
            ],
        ]
        
        opt_sentences = []
        for pidx, prompt in enumerate(prompts):
            opt_prompt = prompt+'[/PROMPT]\n'
            opt_sentence = "[OPTION]".join(options[pidx])
            opt_sentences.append(opt_prompt+opt_sentence)
        bt_opt_sentences = STR2BT(opt_sentences, max_sentence_length=self.max_sentence_length)
        return bt_opt_sentences, gt_answer
    
    def _generate_prompt_options_minmax_vel(
        self, 
        logs,
        rgb='pole',
        minmax='min',
        anglinear='angular',
    ):
        '''
        Generates the prompt and options for the '{pole/cart}_{min/max}_{angular/linear}' QA type.
        WARNING: timestamps are numbered from 1, therefore excluding 0 from possible answers.
        
        :return: a list of strings, each string is a prompt option, along with the groundtruth answer.
        '''
        if rgb=='pole': 
            assert anglinear=='angular'
            velocities = copy.deepcopy(self.nptheta_dots_final)
        if rgb=='cart': 
            assert anglinear=='linear'
            velocities = copy.deepcopy(self.npx_dots_final)
        if minmax=='min': 
            opt_fn = np.min
            argopt_fn = np.argmin
        elif minmax=='max':
            opt_fn = np.max
            argopt_fn = np.argmax
        else:
            raise NotImplementedError

        opt_vel = opt_fn(velocities)
        # GT answer corresponds to the id (from 0) of the option answer that is correct.
        gt_answer = argopt_fn(velocities) # no need to add +1 because we count options id

        if self.notrace:
            assert self.use_cot
            prompt = f"Below is some information about the simulation of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:\n"
            prompt += self.generate_cot(logs, prompt_type=f"{rgb}_{minmax}_{anglinear}")
            prompt += f"You are an expert in the matter. Given the information above, answer the following question to the best of your abilities.\n"
            prompt += "Question: What is the timestamp number where the absolute {anglinear} velocity of the {rgb} reached its {minmax}imum?\n"
            #prompt += "Question: At which timestamp was the {minmax} {anglinear} velocity of the {rgb} reached?\n"
            prompt += "\n"
        else:
            prompt = f"Below is the simulation trace of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:"
            prompt += f"\n\n{logs}\n\n"
            prompt += f"You are an expert in the matter. Given the simulation trace for a cart pole system above, answer the following question to the best of your abilities:\n"
            prompt += "Question: What is the timestamp number where the absolute {anglinear} velocity of the {rgb} reached its {minmax}imum?\n"
            prompt += "\n"
        
            if self.use_cot:
                prompt += self.generate_cot(logs, prompt_type=f"{rgb}_{minmax}_{anglinear}")

        prompts = [ prompt ]
        options = [[
            f'Answer: timestamp {x}.'
            for x in range(1, self.max_nbr_actions[f'{rgb}_{minmax}_{anglinear}']+1) # excluding 0 here
            ],
        ]
        
        opt_sentences = []
        for pidx, prompt in enumerate(prompts):
            opt_prompt = prompt+'[/PROMPT]\n'
            opt_sentence = "[OPTION]".join(options[pidx])
            opt_sentences.append(opt_prompt+opt_sentence)
        bt_opt_sentences = STR2BT(opt_sentences, max_sentence_length=self.max_sentence_length)
        return bt_opt_sentences, gt_answer
    
    def _generate_prompt_options_cart(self, logs):
        '''
        Generates the prompt and options for the 'cart' type.
        WARNING: NOT excluding 0, as opposed to with 'pole' type. 

        :return: a list of strings, each string is a prompt option, along with the groundtruth answer.
        '''
        if self.notrace:
            assert self.use_cot
            prompt = f"Below is some information about the simulation of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:\n"
            prompt += self.generate_cot(logs, prompt_type='cart')
            prompt += f"You are an expert in the matter. Given the information above, answer the following question to the best of your abilities.\n"
            prompt += "Question: How many times did the cart change its direction of translation?\n"
            prompt += "\n"
        else:
            prompt = f"Below is the simulation trace of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:"
            prompt += f"\n\n{logs}\n\n"
            prompt += f"You are an expert in the matter. Given the simulation trace for a cart pole system above, answer the following question to the best of your abilities:\n"
            prompt += "Question: How many times did the cart change its direction of translation?\n"
            prompt += "\n"
        
            if self.use_cot:
                prompt += self.generate_cot(logs, prompt_type='cart')

        prompts = [ prompt ]
        options = [[
            f'Answer: {x} time{"s" if x>1 else ""}.'
            for x in range(self.max_nbr_actions['cart']) # NOT excluding 0 here
            ],
        ]
        
        opt_sentences = []
        for pidx, prompt in enumerate(prompts):
            opt_prompt = prompt+'[/PROMPT]\n'
            opt_sentence = "[OPTION]".join(options[pidx])
            opt_sentences.append(opt_prompt+opt_sentence)
    
        bt_opt_sentences = STR2BT(opt_sentences, max_sentence_length=self.max_sentence_length)
        gt_answer = self.nbr_sliding_changes
        return bt_opt_sentences, gt_answer
    
    def _generate_prompt_options_pole(self, logs):
        '''
        Generates the prompt and options for the 'pole' type.
        WARNING: excluding 0 again from answer choices in 'pole' type. 

        :return: a list of strings, each string is a prompt option, along with the groundtruth answer.
        '''
        if self.notrace:
            assert self.use_cot
            prompt = f"Below is some information about the simulation of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:\n"
            prompt += self.generate_cot(logs, prompt_type='pole')
            prompt += f"You are an expert in the matter. Given the information above, answer the following question to the best of your abilities.\n"
            prompt += "Question: How many times did the pole change its direction of rotation?\n"
            prompt += "\n"
        else:
            prompt = f"Below is the simulation trace of a cart pole/inverted pendulum system,"
            prompt += f" followed by some instructions:"
            prompt += f"\n\n{logs}\n\n"
            prompt += f"You are an expert in the matter. Given the simulation trace for a cart pole system above, answer the following question to the best of your abilities:\n"
            prompt += "Question: How many times did the pole change its direction of rotation?\n"
            prompt += "\n"
        
            if self.use_cot:
                prompt += self.generate_cot(logs, prompt_type='pole')

        prompts = [ prompt ]
        options = [[
            f'Answer: {x} time{"s" if x>1 else ""}.'
            for x in range(1, self.max_nbr_actions['pole']+1) # excluding 0 here
            ],
        ]
        
        opt_sentences = []
        for pidx, prompt in enumerate(prompts):
            opt_prompt = prompt+'[/PROMPT]\n'
            opt_sentence = "[OPTION]".join(options[pidx])
            opt_sentences.append(opt_prompt+opt_sentence)
    
        bt_opt_sentences = STR2BT(opt_sentences, max_sentence_length=self.max_sentence_length)
        gt_answer = self.nbr_rotation_changes-1 # we remove 1 to align with the options id which exclude 0...
        return bt_opt_sentences, gt_answer
    
    def seed(self, seed=None):
        '''
        Update the numpy random number generator with the :param seed:.
        '''
        self.np_random, seed = seeding.np_random(seed)
        self.inverted_pendulum_env.seed(seed)
        return [seed]
 
    def reset(self, **kwargs):
        '''
        WARNING: excluding option 0 only for 'pole' prompt. 
        '''
        if 'seed' in kwargs:
            self.seed(seed=kwargs['seed'])
        # Maximum number of rotation changes are defined with respect to the pole.
        # The cart value is simply a consequence of the pole value.
        desired_nbr_rotation_changes = self.np_random.integers(1, self.max_nbr_actions['pole']+1)
        offset_nbr_rotation_changes = self.np_random.integers(0, self.max_nbr_actions['pole']+1)
        #print(f"Desired nbr rotation changes: {desired_nbr_rotation_changes}, offset nbr rotation changes: {offset_nbr_rotation_changes}")
 
        if kwargs.get('render', False):
            self.inverted_pendulum_env.render(mode='human')
        self.obs, self.info = list(self.inverted_pendulum_env.reset(**kwargs))

        # Collect infos and pendulum's angular velocity:
        infos = []
        theta_dots = []
        x_dots = []
        self.nbr_rotation_changes = 0
        while self.nbr_rotation_changes < desired_nbr_rotation_changes + offset_nbr_rotation_changes:    
            a = np.zeros(self.inverted_pendulum_env.action_space.shape)
            self.obs, reward, done, truncation, info = self.inverted_pendulum_env.step(a)
            if kwargs.get('render', False):
                time.sleep(self.timestep*self.frame_skip)
            theta_dots.append(np.around(self.inverted_pendulum_env.robot.theta_dot, decimals=3)) 
            x_dots.append(np.around(self.inverted_pendulum_env.robot.x_dot, decimals=3)) 
            infos.append(info)
        
            # Compute how many times did the pendulum change direction of rotation: 
            nptheta_dots = np.array(theta_dots)
            self.rotation_change_indices = np.where(nptheta_dots[:-1] * nptheta_dots[1:] < 0)[0]
            self.nbr_rotation_changes = len(self.rotation_change_indices)
             
            # Compute how many times did the cart change direction of sliding: 
            npx_dots = np.array(x_dots)
            self.sliding_change_indices = np.where(npx_dots[:-1] * npx_dots[1:] < 0)[0]
            self.nbr_sliding_changes = len(self.sliding_change_indices)
        
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
        #print(f"start_RC_idx: {start_RC_idx}")
        end_RC_idx = start_RC_idx + desired_nbr_rotation_changes-1 #would cause issue if desired_nbr_rotation_changes=0
        assert end_RC_idx < len(self.rotation_change_indices)
        if start_RC_idx==0 \
        or self.rotation_change_indices[start_RC_idx-1]+1 == self.rotation_change_indices[start_RC_idx]:
            start_iidx = max(self.rotation_change_indices[start_RC_idx]-1, 0)
        else:
            start_iidx = self.np_random.integers(self.rotation_change_indices[start_RC_idx-1]+1, self.rotation_change_indices[start_RC_idx])
        #print(f"Start iidx: {start_iidx}")
        if end_RC_idx==len(self.rotation_change_indices)-1 \
        or self.rotation_change_indices[end_RC_idx]+1 == self.rotation_change_indices[end_RC_idx+1]:
            end_iidx = min(self.rotation_change_indices[end_RC_idx]+1, len(infos))
        else:
            end_iidx = self.np_random.integers(self.rotation_change_indices[end_RC_idx]+1, self.rotation_change_indices[end_RC_idx+1])
        #print(f"End iidx: {end_iidx}")
        
        # Regularisation:
        self.nbr_rotation_changes = desired_nbr_rotation_changes
        self.rotation_change_indices = [idx-start_iidx for idx in self.rotation_change_indices[start_RC_idx:end_RC_idx+1]]
        assert len(self.rotation_change_indices) == desired_nbr_rotation_changes
        self.nptheta_dots_final = np.array(theta_dots[start_iidx:end_iidx+1])
        self.npx_dots_final = np.array(x_dots[start_iidx:end_iidx+1])
        reg_required = False
        self.final_rotation_change_indices = np.where(self.nptheta_dots_final[:-1] * self.nptheta_dots_final[1:] < 0)[0]
        self.final_sliding_change_indices = np.where(self.npx_dots_final[:-1] * self.npx_dots_final[1:] < 0)[0]
        if not self.final_rotation_change_indices.shape[0] == desired_nbr_rotation_changes:
            assert end_iidx-1 > start_iidx
            reg_required = True 
            end_iidx -= 1 # WARNING : cf above
            self.nptheta_dots_final = self.nptheta_dots_final[:-1]
            self.final_rotation_change_indices = self.final_rotation_change_indices[:-1]
            
            self.npx_dots_final = self.npx_dots_final[:-1]
            self.final_sliding_change_indices = self.final_sliding_change_indices[:-1]

        # Check last info contains logs:
        collated_info = {'step2type': {}, 'type2prompt': {}, 'type2gt': {}, 'extras': {}}
        collated_info['extras'] = {'logs': []}
        for info in infos[start_iidx:end_iidx+1]: 
            for log in info['logs']: 
                collated_info['extras']['logs'].append(log)
        collated_info['extras']['log'] = '\n'.join(['\n'.join(l) for l in collated_info['extras']['logs']])
        
        # Add prompt+options:
        self.prompts = {}
        self.groundtruthAnswers = {}
        self.stepIdx2promptType = {}
        for pidx, prompt_type in enumerate(self.prompt_types):
            prompt, groundtruthAnswer = self._generate_prompt_options(
                collated_info['extras']['log'],
                prompt_type=prompt_type,
            )
            self.prompts[prompt_type] = prompt
            self.groundtruthAnswers[prompt_type] = groundtruthAnswer
            self.stepIdx2promptType[pidx] = prompt_type

            collated_info['step2type'][pidx] = prompt_type
            collated_info['type2prompt'][prompt_type] = prompt
            collated_info['type2gt'][prompt_type] = groundtruthAnswer

        self.stepIdx = 0
        collated_info['prompt'] = self.prompts[self.stepIdx2promptType[self.stepIdx]]
        self.info = collated_info
        self.info['predicted_answer'] = -1 
        self.info['groundtruth_answer'] = -1 #self.nbr_rotation_changes 
        self.info['rotation_change_indices'] = self.rotation_change_indices
        self.info['translation_change_indices'] = self.sliding_change_indices
        self.info['step'] = self.stepIdx
         
        action_mask=np.zeros((1, self.action_space.n))
        self.info['action_mask'] = action_mask
        self.info['legal_actions'] = action_mask
        return tuple([self.obs.astype(np.float32), self.info])    

    def step(self, a, **kwargs):
        '''
        The environment has as many steps as there are prompt types.
        Returns +1 reward if the action corresponds to the number of times the pendulum rotated. 
        Else, returns -1.
        '''
        #predicted_nbr_rotation_changes = a+1 # 0 being excluded, answers are at least 1
        predicted_answer = a
        groundtruth_answer = self.groundtruthAnswers[self.stepIdx2promptType[self.stepIdx]]
        
        self.info['predicted_answer'] = predicted_answer
        self.info['groundtruth_answer'] = groundtruth_answer
        
        self.reward = 1 if predicted_answer == groundtruth_answer else -1
        self.stepIdx += 1
        if self.stepIdx >= len(self.prompts): 
            done = True
        else: 
            self.info['prompt'] = self.prompts[self.stepIdx2promptType[self.stepIdx]]
            done = False

        self.info['step'] = self.stepIdx

        return self.obs.astype(np.float32), self.reward, done, False, self.info

    def close(self, **kwargs):
        self.inverted_pendulum_env.close() 

