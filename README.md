# [DIPhiR-Gym](https://github.com/kyd500/DIPhiR-Gym)
An OpenAI Gym environment to investigate Deliberate & Intuitive Physics Reasonings.

At each reset, each environment will randomise some hyperparameters of its related base MuJoCo XML config file, yielding a similar environment with different boundary conditions, such as initial linear/angular velocities of the different rigid bodies, and inertial properties.
The resulting MuJoCo XML config files, along with the phase space data recorded during the simulation, can be stored for later analysis.


## Requirements

* Python>=3.6
* gym>=0.26.2
* numpy 
* pandas
* matplotlib
* Pillow


## Installation

Use the following command to install the environment (with option `-e` for editable installation):

```bash
pip install [-e] .
```

Or manually:

```bash
git submodule update --init --recursive
pip install -r diphirgym/thirdparties/pybulletgym/requirements.txt
python setup.py manual_develop_install
```


## Usage:

For instance, with the inverted pendulum environment, you can run the following code in order to see non-obfuscated logs and have the phase space data along with the randomised MuJoCo XML config files saved to disk in a new `data` folder at your current working directory:

```python
import diphirgym
env = gym.make('InvertedPendulumDIPhiREnv-v0',
    output_dir=os.path.join(os.getcwd(), 'data'),
    obfuscate_logs=False,
)
env.render(mode='human')

state, info = env.reset()
loglist = info['logs']
log = '\n'.join(['\n'.join(l) for l in loglist])
print(log)

for _ in range(256): 
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    
    loglist = info['logs']
    log = '\n'.join(['\n'.join(l) for l in loglist])
    print(log)
    
    if done:
        break
env.close()
```

## Examples & Testing:

Please refer to the example tests in the `tests` folder.


