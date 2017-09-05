from setuptools import setup
from setuptools import Extension

import sys

from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    extensions = [
        Extension('radialprofile', ['radialprofile.pyx'], include_dirs=[np.get_include()])
    ]
else:
    extensions = [
        Extension('radialprofile', ['radialprofile.pyx'], include_dirs=[np.get_include()])
    ]


setup(
    ext_modules=cythonize(extensions)
)
