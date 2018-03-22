#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

import os, sys
from setuptools import setup, Extension

exec(open('pyxem/version.py').read())  # grab version info

from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    extensions = [
        Extension('pyxem.utils.radialprofile', ['src/radialprofile.pyx'], include_dirs=[np.get_include()])
    ]
else:
    extensions = [
        Extension('pyxem.utils.radialprofile', ['src/radialprofile.pyx'], include_dirs=[np.get_include()])
    ]
ext_modules = cythonize(extensions)


setup(
    name='pyxem',
    version=__version__,
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author=__author__,
    author_email=__email__,
    license="GPLv3",
    url="https://github.com/pyxem/pyxem",

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    ext_modules = ext_modules,

    packages=[
        'pyxem',
        'pyxem.utils',
        'pyxem.io_plugins',
	'pyxem.components',
        'pyxem.generators',
	'pyxem.signals'
    ],

#    install_requires=[
#    	'hyperspy',
#        'pymatgen',
#        'transforms3d',
#        'cython',
#        'lxml',
#    ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pyxem": ["*.py"],
    },
)
