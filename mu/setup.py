from setuptools import setup, find_packages

setup(name = 'mu',
    packages = find_packages('.'),
    package_dir = {'mu': 'mu'},
    package_data = {'mu': ['data/*']},
    zip_safe = False)


