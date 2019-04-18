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
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author=__author__,
    author_email=__email__,
    license="GPLv3",
    url="https://github.com/pyxem/pyxem",

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    packages=find_packages(),

    install_requires=[
    	'hyperspy >= 1.3',
        'transforms3d',
	    'scikit-learn >= 0.19'
      ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pyxem": ["*.py"],
    },
)
