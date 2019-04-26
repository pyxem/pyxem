#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from setuptools import setup, find_packages

exec(open('pyxem/version.py').read())  # grab version info


setup(
    name='pyxem',
    version=__version__,
    description='Crystallographic Electron Microscopy in Python.',
    author=__author__,
    author_email=__email__,
    license="GPLv3",
    url="https://github.com/pyxem/pyxem",
    long_description=open('README.rst').read(),
    classifiers=[
	"Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
	"Programming Language :: Python :: 3.7",
	"Development Status :: 4 - Beta",
	"Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    packages=find_packages(),

    install_requires=[
    	'hyperspy >= 1.3',
        'transforms3d',
	'scikit-learn >= 0.19',
        'scikit-image < 0.15'
      ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pyxem": ["*.py"],
    },
)
