Welcome to pixStem's documentation!
===============================================

News
----

**2019-3-5: pixStem 0.3.3 released!**

This release includes:

* Improvements to :ref:`template_match_disk` and :ref:`peak_finding`, so they now work with any number of navigation dimensions, from single diffraction images, line scans and larger. This will be expanded to the other functions in the future, so they will allow for any data dimensionality.
* Improved documentation for the :ref:`fluctuation electron microscopy analysis <fem_analysis>`. Thanks to `Andrew Herzing <https://gitlab.com/aaherzing>`_ for this contribution!
* Several minor bug-fixes.


About pixStem
-------------

Library for processing data acquired on a fast pixelated electron detector, acquired using scanning transmission electron microscopy (STEM).
For more information about this technique and how do analyse this type of data, see the paper `Fast Pixelated Detectors in Scanning Transmission Electron Microscopy. Part I: Data Acquisition, Live Processing and Storage <https://arxiv.org/abs/1911.11560>`_ at arXiv.

Install instructions: :ref:`install`.

pixStem is available under the GNU GPL v3 license, and the source code is found in the `GitLab repository <https://gitlab.com/pixstem/pixstem/tree/master/>`_.

.. image:: images/frontpage/stem_diffraction.jpg
    :scale: 32 %

.. image:: images/frontpage/nbed_example.jpg
    :scale: 32 %

.. image:: images/frontpage/dpc_dummy_data.jpg
    :scale: 32 %


.. toctree::
   install
   start_pixstem
   loading_data
   using_pixelated_stem_class
   analysing_holz_datasets
   analysing_nbed_data
   analysing_dpc_datasets
   fem_analysis
   open_datasets
   generate_test_data
   misc_functions
   related_projects
   api
   :maxdepth: 2
   :caption: Contents:

Old news
--------

*2018-12-3: pixStem 0.3.2 released!*

This release includes:

* Data analysis functionality for analysing :ref:`fluctuation electron microscopy data <fem_analysis>`, thanks to Andrew Herzing for the contribution!
* A new method in the ``PixelatedSTEM`` class for template matching a disk with STEM diffraction data: :ref:`template_match_disk`
* Another method for locating diffraction spots in STEM diffraction data, using peak finding functionality from ``skimage``: :ref:`peak_finding`

All these rely on the `dask <https://dask.org/>`_ library, so they can be performed on very large datasets.


*2018-6-28: pixStem 0.3.1 released!*

This is a minor release which includes functionality for correcting both dead and hot pixels through :py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.find_dead_pixels`, :py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.find_hot_pixels` and :py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.correct_bad_pixels`.
Featurewise it also includes a simple function for loading binary signals: :py:func:`pixstem.io_tools.load_binary_merlin_signal`.
Unit testing of docstrings has been improved, with a combined setup and teardown function, and introduction of pixstem and numpy into the docstring's namespace.


*2018-6-13: pixStem 0.3.0 released!*

fpd_data_processing has been renamed pixStem!
This release also includes greatly improved center of mass, virtual annular dark field and virtual bright field functions, which now uses only `dask <https://dask.pydata.org/en/latest/>`_ array operations.
This means they can return lazy signal, which makes it much easier to work with large datasets on computers with limited memory.
It also includes several backend improvements to ``DPCSignal.correct_ramp`` and utility functions for getting diffraction values.
A Jupyter Notebook giving a basic introduction to pixStem's features is available at `pixStem demos <https://gitlab.com/pixstem/pixstem_demos/blob/master/introduction_to_pixstem.ipynb>`_.


*2018-2-18: fpd_data_processing 0.2.1 released!*

This release includes a major improvement for lazy signal processing with center of mass and radial integration.
It also includes a new method for shifting diffraction patterns in the PixelatedSTEM class.


Acknowledgement
---------------

Initial work from Magnus Nord funded by EPSRC via the project "Fast Pixel Detectors: a paradigm shift in STEM imaging" (Grant reference EP/M009963/1).

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
