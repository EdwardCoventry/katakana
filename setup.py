from setuptools import setup

setup(
    name='katakana',
    version='0.1.1',
    description='English to Katakana with sequence-to-sequence learning',
    license='MIT',
    url='http://github.com/wanasit/katakana',
    author='wanasit',
    packages=['katakana'],
    install_requires=['keras', 'h5py', 'numpy'],
)
