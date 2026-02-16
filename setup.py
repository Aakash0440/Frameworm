from setuptools import setup, find_packages

setup(
    name='frameworm',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'click>=8.0.0',
        'rich>=13.0.0',
        'pyyaml>=6.0',
    ],
    entry_points={
        'console_scripts': [
            'frameworm=frameworm.cli.main:cli',
        ],
    },
)