PyCrystEM (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy.

PyCrystEM builds on the tools for multi-dimensional data analysis provided by the HyperSpy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the PyMatGen library.

PyCrystEM is released under the GPL v3 license.

HEALTH WARNING
--------------

PyCrystEM is in the relatively early stages of development and may contain a number of bugs as a result. The code has been released at this stage in good faith and to speed up this refinement process. We hope that you will find the tools useful and if you find bugs we would appreciate knowing about them.

PyCrystEM only supports Python 3

Install
-------

PyCrystEM can be installed by navigating to the directory containing the package and using pip or running the following command::

	$ python setup.py install

This will not install any of the dependancies, to do so you should add the -r requirements.txt flag. If you do this you will have an environment with a battle-worn set of dependancies. 

New to Python?
--------------

If you are new to python the simplest way to install everything you need is using Anaconda, which can be downloaded from www.continuum.io/downloads (again be aware that you will need Python 3)

From a clean install the following commands to install everything you need should be entered into the terminal, or anaconda prompt terminal in Windows::


	$ conda install hyperspy -c conda-forge

	$ conda install --channel matsci pymatgen

	$ pip install transforms3d

	$ python setup.py install

(Note that conda cannot install pymatgen on a 32-bit machine, use pip)

PyPrismatic
------

This set of installs allows the user to create both conventional multislice and the 'PRISM' algoritm. For details we refer you to http://prism-em.com/

Due to mixed of compiled and non-compiled code some care is required in the install, and we suggest you follow the instructions offered on the website careful. The developers of PyCrystEM note that::

	$ echo $LIBRARY_PATH

Proves very useful in determining whether::

	$ export $LIBRARY_PATH=~/lib/boost_1_65_1

Or similar has been succesful.

Having installed the dependancies (see http://prism-em.com/installation/) one should be able to run::

	$ pip install pyprismatic

without incident. 
