.. _analysing_holz_datasets:

=======================
Analysing HOLZ datasets
=======================

In pixelated scanning transmission electron microscopy (STEM) datasets the higher order laue zone (HOLZ) rings can be used to get information about the structure parallel to the electron beam.
The radius of these rings are proportional to the square root of the size of the crystal's unit cell size.
This guide will show how to extract the size, intensity and width of HOLZ rings.
These parameters can then be used to infer information about the crystal structure of a crystalline material.

Dataset
-------

This example will use a test dataset, containing disk and a ring.
The dataset is found in :py:func:`fpd_data_processing.dummy_data.get_holz_heterostructure_test_signal`
The disk represents the STEM bright field disk, while the ring represents the HOLZ ring.

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.dummy_data.get_holz_heterostructure_test_signal()
    >>> s.plot()

.. image:: images/analysing_holz_datasets/testdata_navigator.png
    :scale: 50 %
    :align: center

.. image:: images/analysing_holz_datasets/testdata_signal.png
    :scale: 50 %
    :align: center

Your own data can be loaded using :py:func:`fpd_data_processing.io_tools.load_fpd_signal`:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.load_fpd_signal(yourfilname)  # doctest: +SKIP
