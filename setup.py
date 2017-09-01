#!/usr/bin/env python

from setuptools import setup


setup(
    name='pycrystem',
    version="0.4",
    description='An open-source Python library for crystallographic electron'
                'microscopy.',
    author="Duncan Johnstone",
    author_email="dnj23@cam.ac.uk",
    packages=[
        'pycrystem',
        'pycrystem.utils',
    ],
    package_data={'pycrystem' : ['utils/atomic_scattering_params.json']}

)
