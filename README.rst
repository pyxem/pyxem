|Actions|_ |Coveralls|_ |pypi_version|_ |downloads|_ |black|_ |doi|_

.. |Actions| image:: https://github.com/pyxem/pyxem/workflows/build/badge.svg
.. _Actions: https://github.com/pyxem/pyxem/actions

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/pyxem/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/pyxem/pyxem?branch=master

.. |pypi_version| image:: http://img.shields.io/pypi/v/pyxem.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/pyxem

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2649351.svg
.. _doi: https://doi.org/10.5281/zenodo.2649351

.. |downloads| image:: https://anaconda.org/conda-forge/pyxem/badges/downloads.svg
.. _downloads: https://anaconda.org/conda-forge/pyxem

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _black: https://github.com/psf/black

pyxem is an open-source python library for multi-dimensional diffraction microscopy.

The package defines objects and functions for the analysis of numerous diffraction patterns. It has been primarily developed as a platform for hybrid diffraction-microscopy based on 4D scanning diffraction microscopy data in which a 2D diffraction pattern is recorded at every position in a 2D scan of a specimen.

pyxem is an extension of the hyperspy library for multi-dimensional data analysis and defines diffraction specific `Signal` classes.

**Installation instructions and tutorial examples are available** `here <https://github.com/pyxem/pyxem-demos>`__ .

**Basic Documentation is available** `here <https://pyxem.readthedocs.io/en/latest/>`__.

If analysis using pyxem forms a part of published work please cite the DOI at the top of this page.
In addition, we would appreciate an additional citation to the following paper if you use the orientation mapping capabilities:

::

    @article{pyxemorientationmapping2022,
        title={Free, flexible and fast: Orientation mapping using the multi-core and GPU-accelerated template matching capabilities in the python-based open source 4D-STEM analysis toolbox Pyxem},
        author={Cautaerts, Niels and Crout, Phillip and {\AA}nes, H{\aa}kon Wiik and Prestat, Eric and Jeong, Jiwon and Dehm, Gerhard and Liebscher, Christian H},
        journal={Ultramicroscopy},
        pages={113517},
        year={2022},
        publisher={Elsevier},
        doi={10.1016/j.ultramic.2022.113517}
    }

pyxem is released under the GPL v3 license.
