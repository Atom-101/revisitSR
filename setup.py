import os
import sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'certifi==2021.5.30',
    'cycler==0.10.0',
    'GPUtil==1.4.0',
    'imageio==2.9.0',
    'kiwisolver==1.3.1',
    'matplotlib==3.4.2',
    'networkx==2.6.2',
    'numpy==1.21.1',
    'olefile==0.46',
    'Pillow==8.3.1',
    'pip==21.2.2',
    'pyparsing==2.4.7',
    'python-dateutil==2.8.2',
    'PyWavelets==1.1.1',
    'PyYAML==5.4.1',
    'scikit-image==0.18.2',
    'scipy==1.7.1',
    'setuptools==52.0.0.post20210125',
    'six==1.16.0',
    'tifffile==2021.7.30',
    'tqdm==4.62.0',
    'typing-extensions==3.10.0.0',
    'wheel==0.36.2',
    'yacs==0.1.8',
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/ptsr'

    setup(name='ptsr',
          description='A PyTorch framework for image super-resolution',
          version=__version__,
          url=url,
          license='MIT',
          author='PTSR Contributors',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
