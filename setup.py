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
        "hyperspy_gui_ipywidgets    >= 2.0",
        "dask-image",
    ],
    "tests": [
        "pytest     >= 5.0",
        "pytest-timeout",
        "pytest-cov >= 2.8.1",
        "pytest-xdist",
        "pytest-rerunfailures",
        "coverage   >= 7.0",
    ],
    "dev": ["black ~= 24.1", "pre-commit >=1.16"],
    "gpu": ["cupy >= 9.0.0", "numba-cuda"],
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
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
        "diffsims       >= 0.7.0",  # ensure we use the latest version
        "hyperspy       >= 2.2.0",  # Use axes_manager set/get functionality
        "h5py",
        "lmfit          >= 1.0.0",  # support for recent version of numpy
        "matplotlib     >= 3.7.5",
        "numba",
        "numpy          >= 1.23.0",  # Needed for numpy.typing
        "orix           >= 0.12.1",
        "pooch",
        "psutil",
        "scikit-image   >= 0.19.0, !=0.21.0, <0.26.0",  # regression in ellipse fitting, EllipseModel issue in 0.26.0
        "scikit-learn   >= 1.1.0",
        "shapely        > 2.0.0",  # major changes
        "scipy",
        "tqdm",
        "traits",
        "transforms3d",
        "zarr            < 3.0.0",  # for fast saving/loading (>=3.0.0 is not compatible)
    ],
    python_requires=">=3.9",
    package_data={
        "": ["LICENSE", "README.rst"],
        "pyxem": ["*.py", "hyperspy_extension.yaml"],
    },
    entry_points={"hyperspy.extensions": ["pyxem = pyxem"]},
)
