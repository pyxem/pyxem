#!/usr/bin/env python

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
#        'cython'
#    ],

    package_data={
        "": ["LICENSE", "readme.rst", "requirements.txt"],
        "pyxem": ["*.py"],
    },
)
