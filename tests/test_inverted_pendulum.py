import pybulletgym
import gym
import numpy as np

def test_logs_inverted_pendulum():
        env = gym.make('InvertedPendulumSwingupPyBulletEnv-v0')
        env.render(mode='human')
        env.reset()
        for _ in range(1000): 
            state, reward, done, truncated, info = env.step(env.action_space.sample())
            loglist = info['logs']
            print('\n'.join(['\n'.join(l) for l in loglist])) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                break #env.reset()
        env.close()

def test_obfuscated_logs_inverted_pendulum():
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

def test_logs_inverted_double_pendulum():
        env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
        env.render(mode='human')
        env.reset()
        for _ in range(1000): 
            state, reward, done, truncated, info = env.step(env.action_space.sample())
            loglist = info['logs']
            print('\n'.join(['\n'.join(l) for l in loglist])) 
            if done:
                print('Episode finished after {} timesteps'.format(env.nbr_time_steps)) 
                #break #env.reset()
        env.close()

def test_obfuscated_logs_inverted_double_pendulum():
        env = gym.make('InvertedDoublePendulumPyBulletEnv-v0', obfuscate_logs=True)
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
                #break #env.reset()
        env.close()

if __name__ == '__main__':
    #test_logs_inverted_pendulum()
    #test_obfuscated_logs_inverted_pendulum()
    test_logs_inverted_double_pendulum()
    #test_obfuscated_logs_inverted_double_pendulum()
