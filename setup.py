#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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
    "doc": [
        "nbsphinx                   >= 0.7",
        "sphinx                     >= 3.0.2",
        "sphinx-copybutton          >= 0.2.5",
        "sphinx-autodoc-typehints   >= 1.10.3",
        "sphinx-gallery             >= 0.6",
        "sphinxcontrib-bibtex       >= 1.0",
        "sphinx_design",
        "sphinx-codeautolink",
        "pydata-sphinx-theme",
        "hyperspy-gui-ipywidgets",
        "dask-image",
    ],
    "tests": [
        "pytest     >= 5.0",
        "pytest-cov >= 2.8.1",
        "pytest-xdist",
        "pytest-rerunfailures",
        "coveralls  >= 1.10",
        "coverage   >= 5.0",
    ],
    "dev": ["black", "pre-commit >=1.16"],
    "gpu": ["cupy >= 9.0.0"],
    "dask": ["dask-image", "distributed"],
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
        "Programming Language :: Python :: 3.10",
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
        "dask",
        "diffsims       >= 0.5",
        "hyperspy       >= 1.7.0, <2.0rc0",  # significant improvements
        "h5py",
        "lmfit          >= 0.9.12",
        "matplotlib     >= 3.3",
        "numba",
        "numpy",
        "numexpr != 2.8.6",  # bug in 2.8.6 for greek letters need for pyfai
        "orix           >= 0.9",
        "psutil",
        "pyfai",  # sigma clip function broken
        "scikit-image   >= 0.19.0, !=0.21.0",  # regression in ellipse fitting"
        "scikit-learn   >= 1.0",
        "scipy",
        "tqdm",
        "traits",
        "transforms3d",
    ],
    python_requires=">=3.7",
    package_data={
        "": ["LICENSE", "README.rst"],
        "pyxem": ["*.py", "hyperspy_extension.yaml"],
    },
    entry_points={"hyperspy.extensions": ["pyxem = pyxem"]},
)
