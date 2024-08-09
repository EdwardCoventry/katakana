from setuptools import setup
import os

setup(
    name='katakana',
    version='0.2.0',
    description='English to Katakana with sequence-to-sequence learning',
    license='MIT',
    url='http://github.com/wanasit/katakana',
    author='wanasit/edward',
    packages=['katakana'],
    package_data={'katakana': ['usemodelconfig.yaml', 'trained_models/**/*']},
    install_requires=['keras', 'h5py', 'numpy'],
)
