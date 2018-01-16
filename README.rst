.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master


pyXem (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy.

pyXem builds on the tools for multi-dimensional data analysis provided by the HyperSpy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the PyMatGen library.

pyXem is released under the GPL v3 license.

HEALTH WARNING
--------------

pyXem is in the relatively early stages of development and may contain a number of bugs as a result. The code has been released at this stage in good faith and to speed up this refinement process. We hope that you will find the tools useful and if you find bugs we would appreciate knowing about them.

pyXem only supports Python 3

Install
-------

pyXem can be installed by navigating to the directory containing the package and using pip or running the following command::

	$ python setup.py install

This will not install any of the dependancies, to do so you should add the -r requirements.txt flag. If you do this you will have an environment with a battle-worn set of dependancies.

New to Python?
--------------

If you are new to python the simplest way to install everything you need is using Anaconda, which can be downloaded from www.continuum.io/downloads (again be aware that you will need Python 3)

From a clean install the following commands to install everything you need should be entered into the terminal, or anaconda prompt terminal in Windows::


	$ conda install hyperspy -c conda-forge

	$ conda install --channel matsci pymatgen Cython

	$ pip install transforms3d

	$ python setup.py install

(Note that conda cannot install pymatgen on a 32-bit machine, use pip)

PyPrismatic
------

This set of installs allows the user to create both conventional multislice and the 'PRISM' algoritm. For details we refer you to http://prism-em.com/

Due to mixed of compiled and non-compiled code some care is required in the install, and we suggest you follow the instructions offered on the website careful. The developers of PyCrystEM note that::

	$ echo $LIBRARY_PATH

Proves very useful in determining whether::

	$ export LIBRARY_PATH=~/lib/boost_1_65_1

Or similar has been succesful.

Having installed the dependancies (see http://prism-em.com/installation/) one should be able to run::

	$ pip install pyprismatic

without incident. 

Citing pyXem
------------

If pyXem has enabled significant parts of an academic publication, please acknowledge that by citing the software. Until a specific publication is written about pyXem please site the github URL: www.github.com/pyxem/pyXem

We also recommend that you cite HyperSpy: http://hyperspy.org/hyperspy-doc/current/citing.html

and PyMatGen:

Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A. Persson, Gerbrand Ceder. Python Materials Genomics (pymatgen) : A Robust, Open-Source Python Library for Materials Analysis. Computational Materials Science, 2013, 68, 314–319. doi:10.1016/j.commatsci.2012.10.028
