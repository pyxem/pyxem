PyCrystEM (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy.

PyCrystEM builds on the tools for multi-dimensional data analysis provided by the HyperSpy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the PyMatGen library.

PyCrystEM is released under the GPL v3 license.

HEALTH WARNING
--------------

PyCrystEM is in the relatively early stages of development and may contain a number of bugs as a result. The code has been released at this stage in good faith and to speed up this refinement process. We hope that you will find the tools useful and if you find bugs we would appreciate knowing about them.


Install
-------

PyCrystEM can be installed by navigating to the directory containing the package and using pip or running the following command:

python setup.py install


New to Python?
--------------

If you are new to python the simplest way to install everything you need is using Anaconda, which can be downloaded from www.continuum.io/downloads

From a clean install the following commands to install everything you need should be entered into the terminal, or anaconda prompt terminal in Windows:

conda install hyperspy -c conda-forge

conda install --channel matsci pymatgen

pip install transforms3d

python setup.py install
