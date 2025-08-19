# setup.py
from setuptools import setup, find_packages

setup(
    name='hps',
    version='1.0',
    packages=['hps'],
    license="MIT",
    author="Joseph Kump, Anna Yesypenko",
    url='https://github.com/annayesy/hps-multidomain-disc',
    install_requires=[
        'numpy','matplotlib','scipy', 'torch', 'pytest'
    ],
)
