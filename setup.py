from setuptools import setup, find_packages


setup(
    name='gym_trainer',
    packages=find_packages('gym_trainer'),
    author='Takuya Aida',
    install_requires=[
        'gym', 'matplotlib', 'torch',
    ]
)