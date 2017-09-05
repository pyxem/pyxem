#!/usr/bin/env python

import os, sys
from setuptools import setup

exec(open('pycrystem/version.py').read())  # grab version info 

setup(
    name='pycrystem',
    version=__version__,
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author=__author__,
    author_email=__email__,
    license="GPLv3",
    url="https://github.com/pycrystem/pycrystem",

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    packages=[
        'pycrystem',
        'pycrystem.utils',
    ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pycrystem": ["*.py", "utils/atomic_scattering_params.json"],
    },
)
