.. _analysing_holz_datasets:

=======================
Analysing HOLZ datasets
=======================

In pixelated scanning transmission electron microscopy (STEM) datasets the higher order laue zone (HOLZ) rings can be used to get information about the structure parallel to the electron beam.
The radius of these rings are proportional to the square root of the size of the crystal's unit cell size.
This guide will show how to extract the size, intensity and width of HOLZ rings.
These parameters can then be used to infer information about the crystal structure of a crystalline material.

Example of this type of data processing: `Three-dimensional subnanoscale imaging of unit cell doubling due to octahedral tilting and cation modulation in strained perovskite thin films <https://doi.org/10.1103/PhysRevMaterials.3.063605>`_.
The data and processing scripts used in this paper is available at Zenodo with the DOI `10.5281/zenodo.3476746 <https://dx.doi.org/10.5281/zenodo.3476746>`_, and Medipix3 dataset can be downloaded directly: https://zenodo.org/record/3476746/files/m004_LSMO_LFO_STO_medipix.hdf5?download=1.

Loading dataset
---------------

This example will use a test dataset, containing disk and a ring.
The dataset is found in :py:func:`pixstem.dummy_data.get_holz_heterostructure_test_signal`
The disk represents the STEM bright field disk, while the ring represents the HOLZ ring.

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.dummy_data.get_holz_heterostructure_test_signal()

Your own data can be loaded using :py:func:`pixstem.io_tools.load_ps_signal`:

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.load_ps_signal(yourfilname)  # doctest: +SKIP

In some cases, the datasets might be too large to load into memory.
For these datasets,  lazy loading can be used.
For more information on this, see :ref:`loading_data`.


Visualizing the data
--------------------

The data is loaded as an :py:class:`~pixstem.pixelated_stem_class.PixelatedSTEM` object, which extends HyperSpy's Signal2D class.
All functions which are present Signal2D is also in the PixelatedSTEM class.

.. code-block:: python

    >>> s
    <PixelatedSTEM, title: , dimensions: (40, 40|80, 80)>

To visualize the dataset, we use:

.. code-block:: python

    >>> s.plot()

.. image:: images/analysing_holz_datasets/testdata_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_signal.png
    :scale: 49 %

Drag the red box (lower right corner in the navigation plot) change the probe position.
See the `HyperSpy documentation <http://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html#multidimensional-spectral-data>`_ for more info on how to use the plotting features.

To change the contrast: select the figure window, and press the H button.

Changing the contrast makes it much easier to see the ring.

.. image:: images/analysing_holz_datasets/testdata_better_contrast_signal.png
    :scale: 49 %


Finding the centre position
---------------------------

To do radial average of the datasets, we first need to find the centre position of the diffraction patterns.
The easiest way of doing this is using :py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.center_of_mass`

.. code-block:: python

    >>> s_com = s.center_of_mass(threshold=2, show_progressbar=False)
    >>> s_com
    <DPCSignal2D, title: , dimensions: (2|40, 40)>
    >>> s_com.plot()

.. image:: images/analysing_holz_datasets/testdata_com_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_com_signal.png
    :scale: 49 %

This returns a :py:class:`~pixstem.pixelated_stem_class.DPCSignal2D` object, which is another specialized class for analysing disk shifts (for example from magnetic materials).
For more information about how to use this for analysing magnetic materials see (TO BE WRITTEN).

The first navigation index is the beam shifts in the x-direction, and the second is the beam shifts in the y-direction.


Doing the radial average
------------------------

The next step is radially integrating the dataset as a function of distance from the centre position, which is done using :py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.radial_average`.

.. code-block:: python

    >>> s_radial = s.radial_average(centre_x=s_com.inav[0].data, centre_y=s_com.inav[1].data, show_progressbar=False)
    >>> s_radial
    <Signal1D, title: , dimensions: (40, 40|62)>
    >>> s_radial.plot()

.. image:: images/analysing_holz_datasets/testdata_radial_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_radial_signal.png
    :scale: 49 %

Now, the ring seen earlier is visible as a peak at x=30 in the signal plot.

A nice way of visualizing this is by transposing the signal, which swaps the signal and navigation axes.
Plot the signal, and move the red line in the navigator plot to x=32.

.. code-block:: python

    >>> s_radial.T.plot()

.. image:: images/analysing_holz_datasets/testdata_radial_T_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_radial_T_signal.png
    :scale: 49 %


Modelling the HOLZ ring
-----------------------

Having reduced the dataset from 4 to 3 dimensions, the HOLZ ring (now a peak, due to the radial average) can easily be fitting with a Gaussian function.

Firstly we extract parts of the signal related to the peak, and create a model.

.. code-block:: python

    >>> s_radial_cropped = s_radial.isig[20:40]
    >>> m_r = s_radial_cropped.create_model()

Due to the noise, the mean value outside the peak is not zero.
To account for this, we fit an offset component to the parts of the signal not containing the peak.
For real datasets, a PowerLaw component should be used (instead of the Offset component).

.. code-block:: python

    >>> from hyperspy.components1d import Offset
    >>> offset = Offset()
    >>> m_r.set_signal_range(20., 25.)
    >>> m_r.set_signal_range(37., 40.)
    >>> m_r.append(offset)
    >>> m_r.multifit(show_progressbar=False)
    >>> m_r.reset_signal_range()
    >>> m_r.plot()

.. image:: images/analysing_holz_datasets/testdata_offset_model_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_offset_model_signal.png
    :scale: 49 %

Then add a Gaussian function to this model.

.. code-block:: python

    >>> from hyperspy.components1d import Gaussian
    >>> g = Gaussian(A=10, centre=30, sigma=4)
    >>> m_r.append(g)
    >>> g.centre.bmin, g.centre.bmax = 25, 35
    >>> m_r.multifit(fitter='mpfit', bounded=True, show_progressbar=False)
    >>> m_r.plot()

.. image:: images/analysing_holz_datasets/testdata_gaussian_model_navigator.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_gaussian_model_signal.png
    :scale: 49 %

The various parameters in the Gaussian can then be visualized.

.. code-block:: python

    >>> g.A.plot()
    >>> g.centre.plot()

.. image:: images/analysing_holz_datasets/testdata_gaussian_amplitude.png
    :scale: 49 %
.. image:: images/analysing_holz_datasets/testdata_gaussian_centre.png
    :scale: 49 %
