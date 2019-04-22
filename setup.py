from setuptools import setup, find_packages


setup(
    name='gym_trainer',
    packages=find_packages('gym_trainer'),
    author='Takuya Aida',
    version='0.1.0',
    install_requires=[
        'gym==0.10.9', 'matplotlib==3.0.2', 'chainer==5.1.0', 'cupy==5.1.0', 'torch==1.0.0', 'torchvision==0.2.1',
        'optuna==0.4.0', 'tensorboardX'
    ]
)