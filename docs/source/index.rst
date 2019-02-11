.. pyxem documentation master file, created by
   sphinx-quickstart on Fri Sep 16 14:34:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
    :hidden:

    self
    literature
    conventions
    contributing


.. figure:: images/forkme_right_orange_ff7600.png
    :align: right
    :target: https://github.com/pyxem/pyxem

pyXem - Crystallographic Electron Microscopy in Python
======================================================

.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

pyXem is an open-source Python library for crystallographic electron microscopy.
The code is primarily developed as a platform for hybrid diffraction-imaging
microscopy based on scanning electron diffraction (SED) data.

.. figure:: images/sped_scheme.png
   :align: center
   :width: 600

pyXem is released under the GPL v3 license.

If analysis using pyxem forms a part of published work please consider recognizing the 
code development by citing the github repository.


Installation
------------

pyXem requires python 3 and conda - we suggest using the python 3 version of `Miniconda <https://conda.io/miniconda.html>`__ and creating a new environment for pyxem using the following commands in the anaconda prompt:::

      $ conda create -n pyxem
      $ conda activate pyxem

Download the `source code <https://github.com/pyxem/pyxem>`__ and put it in a directory on your computer. The following commands will then install everything you need if entered into the anaconda promt (or terminal) when located in the pyxem directory:::

      $ conda install -c conda-forge diffpy.structure
      $ conda install -c anaconda cython
      $ conda install -c conda-forge spglib
      $ conda install -c conda-forge traits
      $ pip install . -r requirements.txt


Getting Started
---------------

To get started using pyxem, especially if you are unfamiliar with python, we recommend using jupyter notebooks. Having installed pyxem as above, a jupyter notebook can be opened using the following commands entered into an anaconda prompt or terminal:::

      $ conda activate pyxem
      $ jupyter notebook

`Tutorials and Example Workflows <https://github.com/pyxem/pyxem-demos>`__ have been curated as a series of jupyter notebooks that you can work through and modify to perform many common analyses.


Documentation
-------------

`Documentation <http://pyxem.github.io/pyxem/pyxem>`__ is available for all pyxem modules at the following links:

- `pyxem.signals <http://pyxem.github.io/pyxem/pyxem.signals>`__ - for manipulating raw data and analysis results.

- `pyxem.generators <http://pyxem.github.io/pyxem/pyxem.generators>`__ - for establishing simulation and analysis conditions.

- `pyxem.components <http://pyxem.github.io/pyxem/pyxem.components>`__ - for fitting in model based analyses.

- `pyxem.libraries <http://pyxem.github.io/pyxem/pyxem.libraries>`__ - that store simulation results needed for analysis.


Questions
---------

If you have a question about pyxem, an issue using the code, or find a bug - we want to know!

We prefer if you let us know by `raising an issue <https://github.com/pyxem/pyxem/issues>`_ 
on our Github page so that we can answer in "public" and potentially help someone else who has
the same question. You can also e-mail the development team via: pyxem.team@gmail.com

Answers may sometimes also be found in our `documentation <http://pyxem.github.io/pyxem/pyxem>`__.


Contributing and Feature Requests
---------------------------------

Feature requests, if pyxem doesn't do something you want it to, can be made by
`raising an issue <https://github.com/pyxem/pyxem/issues>`_ or e-mailing the
development team via pyxem.team@gmail.com

Contributions from new developers are strongly encouraged. Many potential contributors 
may be scientists with little or no open-source development experience and we have written 
a `contributing guide <http://pyxem.github.io/pyxem/contributing.html>`_ particularly for 
this audience.


Related Packages
----------------

The following packages are developed by the pyXem team:

- `texpy <http://pyxem.github.io/pyxem/texpy>`__- for quaternion, rotation, and orientation handling in Python.


These packages are central to the scientific functionality of pyXem:

- `HyperSpy <http://hyperspy.org>`__ for multi-dimensional data handling.

- `DiffPy <http://diffpy.org>`__ - for atomic structure manipulation. 


.. warning::

    The pyXem project is under continual development and there may be bugs. All methods must be used with care.
