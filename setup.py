#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

exec(open("pyxem/release_info.py").read())  # grab version info

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "tests": ["pytest>=5.0", "pytest-cov>=2.8.1", "coveralls>=1.10", "coverage>=5.0"]
}


setup(
    name=name,
    version=version,
    description="multi-dimensional diffraction microscopy",
    author=author,
    author_email=email,
    license=license,
    url="https://github.com/pyxem/pyxem",
    long_description=open("README.rst").read(),
    keywords=[
        "data analysis",
        "diffraction",
        "microscopy",
        "electron diffraction",
        "electron microscopy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    extras_require=extra_feature_requirements,
    install_requires=[
        "scikit-image >= 0.17.0",
        "matplotlib >= 3.1.1",  # 3.1.0 failed
        "scikit-learn >= 0.19",  # reason unknown
        "hyperspy >= 1.6.2",
        "diffsims >= 0.4.0,<0.5",
        "lmfit >= 0.9.12",
        "pyfai",
        "ipywidgets",
        "numba",
        "orix >= 0.3",
    ],
    python_requires=">=3.7",
    package_data={
        "": ["LICENSE", "readme.rst"],
        "pyxem": ["*.py", "hyperspy_extension.yaml"],
    },
    entry_points={"hyperspy.extensions": ["pyxem = pyxem"]},
)
