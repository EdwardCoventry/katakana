from setuptools import setup
import os

setup(
    name='katakana',
    version='0.1.1',
    description='English to Katakana with sequence-to-sequence learning',
    license='MIT',
    url='http://github.com/wanasit/katakana',
    author='wanasit',
    package_data={'katakana': ['usemodelconfig.yaml',
                               os.path.join('trained models', '*')]},
    install_requires=['keras', 'h5py', 'numpy'],
)
