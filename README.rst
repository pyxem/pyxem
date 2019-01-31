.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

.. https://github.com/lemurheavy/coveralls-public/issues/971


Introduction
------------

pyXem (Python Crystallographic Electron Microscopy) is an open-source Python library for crystallographic electron microscopy. It builds on the tools for multi-dimensional data analysis provided by the hyperspy library for treatment of experimental electron diffraction data and tools for atomic structure manipulation provided by the diffpy.Structure library.

pyXem is released under the GPL v3 license.

Install
-------

pyXem requires python 3 and conda (to install a Py3 compatitble version of diffpy) - we suggest using the python 3 version of `Miniconda <https://conda.io/miniconda.html>`__. From a clean install (ideally an isolated anaconda environment) the following commands will install everything you need. They should be entered into the terminal when located in the pyxem directory:::

      $ conda install -c conda-forge diffpy.structure
      $ conda install -c anaconda cython
      $ conda install -c conda-forge spglib
      $ conda install -c conda-forge traits
      $ pip install . -r requirements.txt

NB: This will soon be much simpler once a version of pyXem is uploaded to PyPi and conda, please bear with us in the meantime.

Citing pyXem
------------

If pyXem has enabled significant parts of an academic publication, please acknowledge that by citing the software. Until a specific publication is written about pyXem please cite the github URL: www.github.com/pyxem/pyXem

We also recommend that you cite `HyperSpy <http://hyperspy.org/hyperspy-doc/current/citing.html>`_.

