.. _analysing_dpc_datasets:

==========================================
Analysing differential phase contrast data
==========================================

Differential phase contrast analysis (DPC) is done using using the DPCSignal classes: :py:class:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D`, :py:class:`~fpd_data_processing.pixelated_stem_class.DPCSignal1D` and :py:class:`~fpd_data_processing.pixelated_stem_class.DPCBaseSignal`.

Here, a test dataset is used to show the methods used to process this type of data.
For information on how to load your own data, see `:ref:load_dpc_data`.

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.dummy_data.get_square_dpc_signal(add_ramp=True)
    >>> s.plot()


.. image:: images/analysing_dpc_data/dpc_x_raw.jpg
    :scale: 49 %

.. image:: images/analysing_dpc_data/dpc_y_raw.jpg
    :scale: 49 %


Correcting d-scan
-----------------

The first step is the remove the effects of the d-scan:

.. code-block:: python

    >>> s = s.correct_ramp()
    >>> s.plot()

Note that this method only does a plane correction, using the corners
of the signal. For many datasets, this will not work very well.
Possibly tweaks is to change the corner size.

.. image:: images/analysing_dpc_data/dpc_x_cor.jpg
    :scale: 49 %

.. image:: images/analysing_dpc_data/dpc_y_cor.jpg
    :scale: 49 %


Plotting methods
----------------

Plotting DPC color image using: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.get_color_signal`

.. code-block:: python

    >>> s_color = s.get_color_signal()
    >>> s_color.plot()

.. image:: images/analysing_dpc_data/dpc_color_image.jpg
    :scale: 49 %

Plotting DPC phase image: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.get_phase_signal`

.. code-block:: python

    >>> s_phase = s.get_phase_signal()
    >>> s_phase.plot()

.. image:: images/analysing_dpc_data/dpc_phase_image.jpg
    :scale: 49 %

Plotting DPC magnitude image: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.get_magnitude_signal`

.. code-block:: python

    >>> s_magnitude = s.get_magnitude_signal()
    >>> s_magnitude.plot()

.. image:: images/analysing_dpc_data/dpc_magnitude_image.jpg
    :scale: 49 %

Plotting bivariate histogram: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.get_bivariate_histogram`

.. code-block:: python

    >>> s_hist = s.get_bivariate_histogram()
    >>> s_hist.plot(cmap='viridis')

.. image:: images/analysing_dpc_data/dpc_hist_image.jpg
    :scale: 49 %

Plotting color image with more customizability: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.get_color_image_with_indicator`

.. code-block:: python

    >>> fig = s.get_color_image_with_indicator()
    >>> fig.show()

.. image:: images/analysing_dpc_data/dpc_color_image_indicator.jpg
    :scale: 49 %


Rotating the data
-----------------

Rotating the probe axes: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.rotate_data`.
Note, this will not rotate the beam shifts.

.. code-block:: python

    >>> s_rot_probe = s.rotate_data(10)
    >>> s_rot_probe.get_color_signal().plot()

.. image:: images/analysing_dpc_data/dpc_rotate_probe_color.jpg
    :scale: 49 %

Rotating the beam shifts: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.rotate_beam_shifts`.

.. code-block:: python

    >>> s_rot_shifts = s.rotate_beam_shifts(45)
    >>> s_rot_shifts.get_color_signal().plot()

.. image:: images/analysing_dpc_data/dpc_rotate_shifts_color.jpg
    :scale: 49 %

Rotating both the probe dimensions and beam shifts by 90 degrees: :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.flip_axis_90_degrees`.
Note: in this dataset there will not be any difference compared to the original dataset.
So we slightly alter the dataset.

.. code-block:: python

    >>> s1 = s.deepcopy()
    >>> s1.data[0, 50:250, 145:155] += 5
    >>> s1.get_color_signal().plot()
    >>> s_flip_rot = s1.flip_axis_90_degrees()
    >>> s_flip_rot.get_color_signal().plot()

.. image:: images/analysing_dpc_data/dpc_rotate_flip_color1.jpg
    :scale: 49 %

.. image:: images/analysing_dpc_data/dpc_rotate_flip_color2.jpg
    :scale: 49 %


Blurring the data
-----------------

The beam shifts can be blurred using :py:meth:`~fpd_data_processing.pixelated_stem_class.DPCSignal2D.gaussian_blur`.

This is useful for suppressing the effects of variations in the crystal structure.

.. code-block:: python

    >>> s = fp.dummy_data.get_square_dpc_signal()
    >>> s_blur = s.gaussian_blur()
    >>> s.get_color_signal().plot()
    >>> s_blur.get_color_signal().plot()

.. image:: images/analysing_dpc_data/dpc_gaussian_nonblur.jpg
    :scale: 49 %

.. image:: images/analysing_dpc_data/dpc_gaussian_blur.jpg
    :scale: 49 %
