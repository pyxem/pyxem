.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

.. image:: https://landscape.io/github/pyxem/pyxem/master/landscape.svg?style=flat
   :target: https://landscape.io/github/pyxem/pyxem/master
   :alt: Code Health

.. https://github.com/lemurheavy/coveralls-public/issues/971

Introduction
---

pyXem (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy. It builds on the tools for multi-dimensional data analysis provided by the HyperSpy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the PyMatGen library.

pyXem is released under the GPL v3 license.

Install
-------

pyXem requires python 3 and can be installed by navigating to the directory containing the package and running the following command::

	$ pip install .

However we suggest using an isolated environment. A good option is to use
`Miniconda <https://conda.io/miniconda.html>`__  ( make sure to install the
Python 3 version), which has extensive documentation.

From a clean install the following command will install everything you need. It should be entered into the terminal (located in the pyxem directory)::

	$ pip install . -r requirements.txt

NB: This will soon be much simpler once a version of pyXem is uploaded to PyPi, please bear with us in the meantime.

Citing pyXem
------------

If pyXem has enabled significant parts of an academic publication, please acknowledge that by citing the software. Until a specific publication is written about pyXem please cite the github URL: www.github.com/pyxem/pyXem

We also recommend that you cite `HyperSpy <http://hyperspy.org/hyperspy-doc/current/citing.html>`__
and `PyMatGen <http://pymatgen.org/#how-to-cite-pymatgen>`__
