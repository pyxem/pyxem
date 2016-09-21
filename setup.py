#!/usr/bin/env python

from distutils.core import setup
from pycrystem import __version__, __author__, __email__


setup(
    name='pycrystem',
    version=__version__,
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author=__author__,
    author_email=__email__,
    packages=[
        'pycrystem',
        'pycrystem.utils',
    ],

)
