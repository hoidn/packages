#from distutils.core import setup
from setuptools import setup 

setup(name='mu',
    version='1.0',
    #package_dir = {'': '.'},
    package_data = {'mu': 'data/*'},
    py_modules=['mu', 'edgelookup'],
    #packages = ['']
    )
