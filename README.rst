.. image:: https://travis-ci.org/pyxem/pyxem.svg?branch=master
    :target: https://travis-ci.org/pyxem/pyxem

.. image:: https://ci.appveyor.com/api/projects/status/github/pyxem/pyxem?svg=true&branch=master
    :target: https://ci.appveyor.com/project/dnjohnstone/pyxem/branch/master

.. image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
    :target: https://coveralls.io/github/pyxem/pyxem?branch=master

.. https://github.com/lemurheavy/coveralls-public/issues/971


Introduction
------------

pyXem builds heavily on the tools for multi-dimensional data analysis provided
by the `HyperSpy <http://hyperspy.org>`__ library and draws on `DiffPy <http://diffpy.org>`__
for atomic structure manipulation.

pyXem is released under the GPL v3 license. 

If analysis using pyxem forms a part of published work please consider recognising the code 
development by citing the github repository www.github.com/pyxem/pyXem.

Installation
------------

pyXem requires python 3 and conda - we suggest using the python 3 version of `Miniconda <https://conda.io/miniconda.html>`__. and creating a new environment for pyxem using the following commands in the anaconda prompt:::

      $ conda create -n pyxem
      $ conda activate pyxem

Download the `source code <https://github.com/pyxem/pyxem>`__ and put it in a directory on your computer. The following commands will then install everything you need if entered into the anaconda promt (or terminal) when located in the pyxem directory:::

      $ conda install -c conda-forge diffpy.structure
      $ pip install . -r requirements.txt


Getting Started
---------------

To get started using pyxem, especially if you are unfamiliar with python, we recommend using jupyter notebooks. Having installed pyxem as above, a jupyter notebook can be opened using the following commands entered into an anaconda prompt or terminal:::

      $ conda activate pyxem
      $ jupyter notebook

`Tutorials and Example Workflows <https://github.com/pyxem/pyxem-demos>`__ have been curated as a series of jupyter notebooks that you can work through and modify to perform many common analyses.


`Documentation <http://pyxem.github.io/pyxem>`__ is available via the website.

