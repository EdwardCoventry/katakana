from setuptools import setup

setup(
    name='katakana',
    version='0.1.1',
    description='English to Katakana with sequence-to-sequence learning',
    license='MIT',
    url='http://github.com/wanasit/katakana',
    author='wanasit',
    package_data={'katakana': ['usemodelconfig.yaml']},
    package_data={'': ['usemodelconfig.yaml']},
    install_requires=['keras', 'h5py', 'numpy'],
)
