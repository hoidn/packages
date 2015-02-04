from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages

setup(name='atomicform',
      version='1.0',
      py_modules=['atomicform'],
      package_data={
            'atomicform': ['data/*'],
         },
      )

