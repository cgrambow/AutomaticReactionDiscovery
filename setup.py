#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from distutils.core import setup

###############################################################################

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

modules = []
for root, dirs, files in os.walk('ard'):
    for f in files:
        if f.endswith('.py') or f.endswith('.pyx'):
            if 'Test' not in f and '__init__' not in f:
                module = 'ard' + root.partition('ard')[-1].replace('/', '.') + '.' + f.partition('.py')[0]
                modules.append(module)

setup(
    name='ARD',
    version=0.1,
    description='Automatic Reaction Discovery',
    long_description=readme,
    author='William H. Green and Colin Grambow',
    author_email='cgrambow@mit.edu',
    url='https://github.com/cgrambow/AutomaticReactionDiscovery',
    license=license,
    packages=['ard'],
    py_modules=modules
)
