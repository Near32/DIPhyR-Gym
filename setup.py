import os
from os import path
from io import open
import subprocess
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

def read_requirements(file_path):
    '''
    Read requirements from file
    '''
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

def get_submodules_requirements():
    '''
    Get requirements from submodules (thirdparties) of the repo.
    ''' 
    pybulletgym_requirements = read_requirements('./diphyrgym/thirdparties/pybulletgym/requirements.txt')
    return list(pybulletgym_requirements)

class PreInstallCommand(install):
    '''
    commands to run before installing dependencies.
    '''
    def run(self):
        '''
        install the submodules requirements.  
        '''
        self.execute_submodule_commands()
        install.run(self)

class PreManualDevInstallCommand(develop):
    '''
    commands to run before installing dependencies.
    '''
    def run(self):
        '''
        install requirements manually. 
        '''
        #self.execute_submodule_commands()
        develop.run(self)

class PreDevelopCommand(develop):
    '''
    commands to run before installing dependinces for development.
    '''
    def run(self):
        '''
        install the submodules requirements. 
        ''' 
        self.execute_submodule_commands()
        develop.run(self)

def execute_submodule_commands(self):
    '''
    init and update submodules.
    '''
    subprocess.check_call(['git', 'submodule', 'init'])
    subprocess.check_call(['git', 'submodule', 'update'])

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()


setup(
    name='DIPhyR-Gym',
    version='0.1',
    description='An OpenAI Gym environment to investigate Deliberate & Intuitive Physics Reasonings.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Near32/DIPhyR-Gym',
    author='Kevin Denamganai',
    author_email='denamganai.kevin@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Reinforcement Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    cmdclass={
        'install': PreInstallCommand,
        'manual_develop_install': PreManualDevInstallCommand,
        'develop': PreDevelopCommand,
    },
    zip_safe=False,

    install_requires=[
      'gym==0.26.2',
      'numpy',
      'pandas',
      'matplotlib',
      'Pillow',
      'pre-commit',
    ]+get_submodules_requirements(),

    python_requires=">=3.6",
)

