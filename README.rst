.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

.. image:: https://landscape.io/github/pyxem/pyxem/master/landscape.svg?style=flat
   :target: https://landscape.io/github/pyxem/pyxem/master
   :alt: Code Health

.. https://github.com/lemurheavy/coveralls-public/issues/971


Introduction
------------

pyXem (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy. It builds on the tools for multi-dimensional data analysis provided by the hyperspy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the diffpy.Structure library.

pyXem is released under the GPL v3 license.

Install
-------

pyXem requires python 3 and conda (to install a Py3 compatitble version of diffpy) - as such we suggest using `Miniconda <https://conda.io/miniconda.html>`__  (make sure to install the Python 3 version), which has extensive documentation. From a clean install (ideally an isolated environment) the following commands will install everything you need. They should be entered into the terminal (located in the pyxem directory)::

    | $ conda install -c diffpy/label/dev diffpy.structure 
    | $ pip install . -r requirements.txt

NB: This will soon be much simpler once a version of pyXem is uploaded to PyPi and conda, please bear with us in the meantime.

Citing pyXem
------------

If pyXem has enabled significant parts of an academic publication, please acknowledge that by citing the software. Until a specific publication is written about pyXem please cite the github URL: www.github.com/pyxem/pyXem

We also recommend that you cite `HyperSpy <http://hyperspy.org/hyperspy-doc/current/citing.html>`_.

Publication using pyXem
------------

A number of publications are appearing in the literature using pyXem. Please contact the developers if you wish to be included on this list

at `ChemRVix <https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/7093862/Metal-Organic_Framework_Crystal-Glass_Composites_v1.pdf>`_

at https://www.sciencedirect.com/science/article/pii/S0022024818300617?via%3Dihub

at https://www.sciencedirect.com/science/article/pii/S1044580318304686?via%3Dihub
