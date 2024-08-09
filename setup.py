from setuptools import setup, find_packages
import os

setup(
    name='katakana',
    version='0.2.0',
    description='English to Katakana with sequence-to-sequence learning',
    license='MIT',
    url='http://github.com/wanasit/katakana',
    author='wanasit/edward',
    packages=find_packages(),  # Automatically find and include all packages and subpackages
    package_data={'katakana': ['usemodelconfig.yaml', 'trained_models/**/*']},
    install_requires=['keras', 'h5py', 'numpy'],
)
