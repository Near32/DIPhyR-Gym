import diphirgym
import diphirgym.thirdparties.pybulletgym
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

def test_logs_diphir_inverted_pendulum():
        env = gym.make('InvertedPendulumDIPhiREnv-v0')
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

def test_logs_diphir_inverted_double_pendulum():
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('InvertedDoublePendulumDIPhiREnv-v0',obfuscate_logs=False)
        env.render(mode='human')
        env.reset()
        for _ in range(250): 
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

def test_logs_inverted_double_pendulum():
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('InvertedDoublePendulumPyBulletEnv-v0', obfuscate_logs=False)
        env.render(mode='human')
        env.reset()
        for _ in range(250): 
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

def test_obfuscated_logs_inverted_double_pendulum():
        # Open a file for logging
        log_file = open("simulation_trace.log", "w")

        env = gym.make('InvertedDoublePendulumPyBulletEnv-v0', obfuscate_logs=True)
        env.render(mode='human')
        env.reset()
        for _ in range(250): 
            #action = np.zeros(env.action_space.shape) 
            action = env.action_space.sample()
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
    #test_logs_inverted_pendulum()
    #test_logs_diphir_inverted_pendulum()
    #test_obfuscated_logs_inverted_pendulum()
    test_logs_diphir_inverted_double_pendulum()
    #test_logs_inverted_double_pendulum()
    #test_obfuscated_logs_inverted_double_pendulum()
