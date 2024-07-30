import diphyrgym
import diphyrgym.thirdparties.pybulletgym
import os
import gym
import numpy as np
from tqdm import tqdm


def test_logs_inverted_pendulum():
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
            'max_nbr_actions':5, #10,
            # DEPRECATED: max_nbr_timesteps=16,
            'timestep':0.0165,
            'frame_skip':16,
            'max_sentence_length':1024, #16384,
            #output_dir='/run/user/1000/DIPhiR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            'output_dir':'/run/user/{uid}/DIPhiR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            #output_dir=os.path.join(os.getcwd(), 'data'),
            'obfuscate_logs':False,
            'show_phase_space_diagram':False,
            'save_metadata':False,
            'minimal_logs':False,
            #'minimal_logs':True,
        }
        env = gym.make('OfflineInvertedPendulumDIPhiREnv-v0',
            **config,
        ) 
        # Fixing the seed:
        import wandb
        wandb.init(project="diphyr-tests", config=config)
        gt_answers = []
        prompt_sizes = []
        seed = 0 
        for _ in tqdm(range(100)): #while True:
          seed += 1
          state, info = env.reset(**{'seed': seed})
           
          '''loglist = info['extras']['logs']
          log = '\n'.join(['\n'.join(l) for l in loglist])
          #print(log)
          log_file.write(log) 
        
          print(f"Prompt BT shape is {info['prompt'].shape}")
          '''
          action = np.zeros(env.action_space.shape) 
          #action = env.action_space.sample()
          state, reward, done, truncated, info = env.step(action)
          gt_answers.append(info['groundtruth_answer'])
          prompt_sizes.append(info['prompt'].shape[-1])

          print(f"Predicted vs groundtruth number of rotation changes: {info['predicted_answer']} vs {info['groundtruth_answer']}")
          print(f'Angular velocity change timesteps: {info["rotation_change_indices"]}')
          print('Environment reward: ', reward) 
          #wandb.log({"reward": reward})
        env.close()
        
        # Plot histogram: 
        import matplotlib.pyplot as plt
        plt.hist(gt_answers)
        plt.title(f"Labels Distribution for FrameSkip={config['frame_skip']} & MinimalLogs={config['minimal_logs']}")
        # Save histogram in WandB:
        wandb.log({"histogram_labels": wandb.Image(plt)}) #wandb.Histogram(gt_answers)})
        plt.show()
        
        # Plot histogram: 
        plt.hist(prompt_sizes)
        plt.title(f"Prompt size in bytes for FrameSkip={config['frame_skip']} & MinimalLogs={config['minimal_logs']}")
        # Save histogram in WandB:
        wandb.log({"histogram_prompt_sizes": wandb.Image(plt)}) #wandb.Histogram(prompt_sizes)})
        plt.show()

        # Close the file
        log_file.close()

def test_resets_diphyr_offline_inverted_pendulum():
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhiREnv-v0',
            max_nbr_actions=5, #10,
            # DEPRECATED: max_nbr_timesteps=16,
            timestep=0.0165,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            #output_dir='/run/user/1000/DIPhiR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
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
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhiREnv-v0',
            max_nbr_actions=5, #10,
            # DEPRECATED: max_nbr_timesteps=16,
            timestep=0.0165,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            #output_dir='/run/user/1000/DIPhiR/inverted_pendulum', #os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=True,
            save_metadata=False,
            minimal_logs=True,
        )
        # Fixing the seed:
        state, info = env.reset(**{'seed': 32})
         
        loglist = info['extras']['logs']
        log = '\n'.join(['\n'.join(l) for l in loglist])
        #print(log)
        log_file.write(log) 
        
        print(f"Prompt BT shape is {info['prompt'].shape}")
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
    #test_obfuscated_logs_inverted_pendulum()
