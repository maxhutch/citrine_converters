# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='citrine_converters',
    version='0.1.0',
    description='X-to-PIF converters',
    long_description=readme,
    author='Branden Kappes',
    author_email='bkappes@mines.edu',
    url='https://github.com/csm-adapt/citrine_converters',
    license=license,
    packages=find_packages(exclude=('tests*', 'docs')),

)
