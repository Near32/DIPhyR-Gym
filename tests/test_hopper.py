'''
Licence.
'''
import diphyrgym
import diphyrgym.thirdparties.pybulletgym
import os
import gym
import numpy as np

def test_logs_hopper():
        '''
        Run a single simulation and print the logs.
        '''
        env = gym.make('HopperPyBulletEnv-v0')
        env.render(mode='human')
        env.reset()
        for _ in range(2500): 
            action = np.zeros(env.action_space.shape) 
            state, reward, done, truncated, info = env.step(action)
            loglist = info['logs']
            print('\n'.join(['\n'.join(l) for l in loglist])) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                break #env.reset()
        env.close()

def test_logs_diphyr_hopper():
        '''
        Run a single simulation and print the logs.
        '''
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('HopperDIPhyREnv-v0',
            output_dir=os.path.join(os.getcwd(), 'data'),
            obfuscate_logs=False,
            show_phase_space_diagram=True,#False,
        )
        env.render(mode='human')
        # Fixing the seed:
        state, info = env.reset(**{'seed': 0})
        loglist = info['logs']
        log = '\n'.join(['\n'.join(l) for l in loglist])
        print(log)
        log_file.write(log) 
        for _ in range(256): 
            action = np.zeros(env.action_space.shape) 
            #action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            loglist = info['logs']
            log = '\n'.join(['\n'.join(l) for l in loglist])
            print(log)
            log_file.write(log) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                #break #env.reset()
        env.close()

        # Close the file
        log_file.close()

if __name__ == '__main__':
    #test_logs_hopper()
    test_logs_diphyr_hopper()
