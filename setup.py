#!/usr/bin/env python

from distutils.core import setup
from pycrystem import __version__


setup(
    name='pycrystem',
    version=__version__,
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author='Duncan Johnstone',
    author_email='dnj23@cam.ac.uk',
    packages=[
        'pycrystem',
        'pycrystem.diffraction_generator',
        'pycrystem.diffraction_signal',
        'pycrystem.expt_utils',
        'pycrystem.sim_utils',
    ],

)
