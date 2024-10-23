'''
Licence.
'''
import diphyrgym
import diphyrgym.thirdparties.pybulletgym
import os
import gym
import numpy as np
from tqdm import tqdm

from diphyrgym.utils import STR2BT, BT2STR


def test_logs_inverted_pendulum():
        '''
        Run a single simulation and print the logs.
        '''
        env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')
        env.render(mode='human')
        env.reset()
        for _ in range(250): 
            action = np.zeros(env.action_space.shape) 
            state, reward, done, truncated, info = env.step(action)
            loglist = info['logs']
            print('\n'.join(['\n'.join(l) for l in loglist])) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                break #env.reset()
        env.close()

def test_distr_diphyr_offline_inverted_pendulum():
        '''
        Run multiple simulations and provides a histogram of the groundtruth_answer.
        '''
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        config = {
            #'max_nbr_actions':{'pole': 5, 'cart': 20}, #5, 
            'prompt_types':[
                #'pole', 
                #'pole_min_angular',
                'pole_longest_time',
                'pole_shortest_time',
                #'cart_longest_time',
                #'cart_shortest_time',
                #'cart': 20,
            ], 
            'max_nbr_actions':{
                'pole': 5, 
                'pole_min_angular': 5*8*int(24/16),
                'pole_longest_time': 5*int(24/16),
                'pole_shortest_time': 5*int(24/16),
                'cart_longest_time': 5*int(24/16),
                'cart_shortest_time': 5*int(24/16),
                #'cart': 20,
            }, #5, 
            # DEPRECATED: max_nbr_timesteps=16,
            'timestep':0.0165,
            #'frame_skip':24,#16,
            'frame_skip':16, #4
            'max_sentence_length':1024, #16384,
            #output_dir='/run/user/1000/DIPhyR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            'output_dir':'/run/user/{uid}/DIPhyR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            #output_dir=os.path.join(os.getcwd(), 'data'),
            'obfuscate_logs':False,
            'show_phase_space_diagram':False,
            'save_metadata':False,
            #'minimal_logs':False,
            'minimal_logs':True,
            'use_cot':True,
            'notrace':False,
        }
        env = gym.make('OfflineInvertedPendulumDIPhyREnv-v0',
            **config,
        ) 
        # Fixing the seed:
        import wandb
        wandb.init(project="diphyr-tests", config=config)
        gt_answers = {}
        prompt_sizes = {}
        seed = 0 
        nbr_sims = 1
        done = True
        state, info = env.reset(**{'seed': seed})
        question_id = info['step']
        question_ids = [question_id]
        for _ in tqdm(range(100)): #while True:
          #print(BT2STR(info['prompt'])[0])
          #import ipdb; ipdb.set_trace()
          if question_id not in prompt_sizes: prompt_sizes[question_id] = []
          prompt_sizes[question_id].append(info['prompt'].shape[-1])

          print(f'Angular velocity change timesteps: {info["rotation_change_indices"]}')
          print(f'Linear velocity change timesteps: {info["translation_change_indices"]}')
        
          #action = np.zeros(env.action_space.shape) 
          action = env.action_space.sample()
          state, reward, done, truncated, info = env.step(action)
          if question_id not in gt_answers: gt_answers[question_id] = []
          gt_answers[question_id].append(info['groundtruth_answer'])
          print(f"Previous setep: Predicted vs groundtruth answers: {info['predicted_answer']} vs {info['groundtruth_answer']}")
         
          if done:
            seed += 1
            nbr_sims += 1
            state, info = env.reset(**{'seed': seed})
          question_id = info['step']
          question_ids.append(question_id)
 
        env.close()
        print(f"Nbr Simulations ran: {nbr_sims}") 
        # Plot histogram: 
        import matplotlib.pyplot as plt
        for question_id in range(len(set(question_ids))):
          plt.hist(gt_answers[question_id])
          plt.title(f"Question{question_id} : Labels Distribution for FrameSkip={config['frame_skip']} & MinimalLogs={config['minimal_logs']}")
          # Save histogram in WandB:
          wandb.log({f"histogram_labels/Q{question_id}": wandb.Image(plt)}) #wandb.Histogram(gt_answers)})
          plt.show()
        
          # Plot histogram: 
          plt.hist(prompt_sizes[question_id])
          plt.title(f"Question{question_id} : Prompt size in bytes for FrameSkip={config['frame_skip']} & MinimalLogs={config['minimal_logs']}")
          # Save histogram in WandB:
          wandb.log({f"histogram_prompt_sizes/Q{question_id}": wandb.Image(plt)}) #wandb.Histogram(prompt_sizes)})
          plt.show()

        # Close the file
        log_file.close()

def test_resets_diphyr_offline_inverted_pendulum():
        '''
        Investigate wheter the reset function takes the seed into account and its time complexity.
        Its time complexity is affected by the output_dir parameter.
        It is important to use a path under the tmp folder.
        '''
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhyREnv-v0',
            max_nbr_actions={'pole': 5, 'cart': 20}, #5, 
            # DEPRECATED: max_nbr_timesteps=16,
            timestep=0.0165,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            #output_dir='/run/user/1000/DIPhyR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            #output_dir=os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=False,
            save_metadata=False,
            minimal_logs=True,
        )
        # Fixing the seed:
        import wandb
        wandb.init(project="diphyr-tests")
        for _ in tqdm(range(500)): #while True:
          state, info = env.reset(**{'seed': 32})
           
          '''loglist = info['extras']['logs']
          log = '\n'.join(['\n'.join(l) for l in loglist])
          #print(log)
          log_file.write(log) 
        
          print(f"Prompt BT shape is {info['prompt'].shape}")
          '''
          action = np.zeros(env.action_space.shape) 
          #action = env.action_space.sample()
          state, reward, done, truncated, info = env.step(action)
          print(f"Predicted vs groundtruth number of rotation changes: {info['predicted_answer']} vs {info['groundtruth_answer']}")
          print(f'Angular velocity change timesteps: {info["rotation_change_indices"]}')
          print('Environment reward: ', reward) 
          wandb.log({"reward": reward})
        env.close()

        # Close the file
        log_file.close()

def test_logs_diphyr_offline_inverted_pendulum():
        '''
        Investigate logging.
        '''
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhyREnv-v0',
            max_nbr_actions={'pole': 5, 'cart': 20}, #5, 
            #max_nbr_actions=20,
            # DEPRECATED: max_nbr_timesteps=16,
            timestep=0.00165,
            #frame_skip=2,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            #output_dir='/run/user/1000/DIPhyR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=True,
            save_metadata=False,
            minimal_logs=True,
        )
        # Fixing the seed:
        #state, info = env.reset(**{'seed': 32})
        # DEBUG: with renderring: state, info = env.reset(**{'seed': 33, 'render': True})
        state, info = env.reset(**{'seed': 33, 'render': False})
         
        loglist = info['extras']['logs']
        log = '\n'.join(['\n'.join(l) for l in loglist])
        #print(log)
        log_file.write(log) 
        
        print(f"Prompt BT shape is {info['prompt'].shape}")
        print(f'Angular velocity change timesteps: {info["rotation_change_indices"]}')
         
        action = np.zeros(env.action_space.shape) 
        #action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        
        loglist = info['extras']['logs']
        log = '\n'.join(['\n'.join(l) for l in loglist])
        #print(log)
        log_file.write(log) 
        
        print(f"Prompt BT shape is {info['prompt'].shape}")
        
        print(f"Previous step: Predicted vs groundtruth number of rotation changes: {info['predicted_answer']} vs {info['groundtruth_answer']}")
        print(f'Linear velocity change timesteps: {info["translation_change_indices"]}')
        print('Environment reward: ', reward) 
        
        action = np.zeros(env.action_space.shape) 
        #action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"Previous step: Predicted vs groundtruth number of translation changes: {info['predicted_answer']} vs {info['groundtruth_answer']}")
        
        env.close()

        # Close the file
        log_file.close()

def test_cot_diphyr_offline_inverted_pendulum():
        '''
        Investigates using CoT in the offline environment.
        '''
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhyREnv-v0',
            max_nbr_actions={'pole': 5, 'cart': 20}, #5, 
            # DEPRECATED: max_nbr_timesteps=16,
            timestep=0.0165,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            #output_dir='/run/user/1000/DIPhyR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=True,
            save_metadata=False,
            minimal_logs=True,
            #minimal_logs=False,
            use_cot=True,
        )
        # Fixing the seed:
        state, info = env.reset(**{'seed': 32})
         
        loglist = info['extras']['logs']
        log = '\n'.join(['\n'.join(l) for l in loglist])
        #print(log)
        log_file.write(log) 
        
        print(f"Prompt BT shape is {info['prompt'].shape}")
        print(BT2STR(info['prompt']))
        
        action = np.zeros(env.action_space.shape) 
        #action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"Predicted vs groundtruth number of rotation changes: {info['predicted_answer']} vs {info['groundtruth_answer']}")
        print(f'Angular velocity change timesteps: {info["rotation_change_indices"]}')
        print('Environment reward: ', reward) 
        env.close()

        # Close the file
        log_file.close()


def test_obfuscated_logs_inverted_pendulum():
        '''
        Investigates obfuscation of the logs and prompts.
        '''
        raise NotImplementedError
        env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0', obfuscate_logs=True)
        env.render(mode='human')
        env.reset()
        for _ in range(1000): 
            #action = np.zeros(env.action_space.shape) 
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            loglist = info['logs']
            print('\n'.join(['\n'.join(l) for l in loglist])) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                break #env.reset()
        env.close()

if __name__ == '__main__':
    #test_logs_inverted_pendulum()
    #test_logs_diphyr_offline_inverted_pendulum()
    #test_resets_diphyr_offline_inverted_pendulum()
    test_distr_diphyr_offline_inverted_pendulum()
    #test_cot_diphyr_offline_inverted_pendulum()
    #test_obfuscated_logs_inverted_pendulum()
