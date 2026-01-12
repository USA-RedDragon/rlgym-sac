from setuptools import setup, find_packages
from setuptools.command.install import install


__version__ = "1.3.13"

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()


class CustomInstall(install):
    def run(self):
        install.run(self)

setup(
    name='rlgym-sac',
    packages=find_packages(),
    version=__version__,
    description='A multi-processed implementation of SAC for use with RLGym.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jacob McSwain',
    url='https://github.com/USA-RedDragon/rlgym-sac',

    install_requires=[
        'gymnasium>=1.2.3',
        'numpy>=1.21',
        'rlviser-py>=0.6.13',
        'rocketsim>=2.1.1.post4',
        'torch>=1.13',
        'wandb>=0.15',
    ],
    python_requires='>=3.7',
    cmdclass={'install': CustomInstall},
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gymnasium', 'reinforcement-learning', 'simulation', 'sac', 'rlgym', 'rocketsim']
)
