import diphyrgym
import diphyrgym.thirdparties.pybulletgym
import os
import gym
import numpy as np

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

def test_logs_diphyr_offline_inverted_pendulum():
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('OfflineInvertedPendulumDIPhiREnv-v0',
            max_nbr_actions=5, #10,
            max_nbr_timesteps=16,
            timestep=0.0165,
            frame_skip=24,
            max_sentence_length=1024, #16384,
            output_dir=os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=True,
            minimal_logs=True,
        )
        # Fixing the seed:
        state, info = env.reset(**{'seed': 132})
        
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
    test_logs_diphyr_offline_inverted_pendulum()
    #test_obfuscated_logs_inverted_pendulum()
