from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='DPhiR-Gym',
    version='0.1',
    description='An OpenAI Gym environment to investigate Deliberate & Intuitive Physics Reasonings.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Near32/DPhiR-Gym',
    author='Kevin Denamganai',
    author_email='denamganai.kevin@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Reinforcement Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=[
      'gym==0.26.2',
    ],

    python_requires=">=3.6",
)

