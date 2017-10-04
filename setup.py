#!/usr/bin/env python

import os, sys
from setuptools import setup, Extension

exec(open('pycrystem/version.py').read())  # grab version info 

from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    extensions = [
        Extension('pycrystem.utils.radialprofile', ['src/radialprofile.pyx'], include_dirs=[np.get_include()])
    ]
else:
    extensions = [
        Extension('pycrystem.utils.radialprofile', ['src/radialprofile.pyx'], include_dirs=[np.get_include()])
    ]
ext_modules = cythonize(extensions)


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

    ext_modules = ext_modules,

    packages=[
        'pycrystem',
        'pycrystem.utils',
    ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pycrystem": ["*.py"],
    },
)
