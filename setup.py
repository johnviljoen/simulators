"""
generate a python env with version 3.10, and pip install -e . on this and you
should be good to go.

ensure the mujoco binary is also installed onto the system you are running onto in the proper way:
https://github.com/google-deepmind/mujoco
"""

from setuptools import setup, find_packages

setup(name='simulators',
    version='0.0.1',
    # url='https://github.com/pnnl/dpc_for_robotics/',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'torch',        # standard pytorch pip install
        'torchvision',  # standard pytorch pip install
        'torchaudio',   # standard pytorch pip install
        'casadi',       # need some simulators in casadi
        'matplotlib',   # plotting...
        'mujoco',       # for some simulators - case by case
        'tqdm',         # just for pretty loops in a couple places
    ],
    packages=find_packages(
        include=[
            'dynamics.*',
            'visualisers.*'
        ]
    ),
)
