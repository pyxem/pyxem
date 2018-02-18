.. _using_pixelated_stem_class:

==============================
Using the pixelated STEM class
==============================

The :py:class:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM` class extends HyperSpy's Signal2D class, and makes heavy use of the lazy loading and ``map`` method to do lazy processing.


Visualizing the data
--------------------

If you have a small dataset, ``s.plot`` can be used directly:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.dummy_data.get_holz_simple_test_signal()
    >>> s.plot()

If the dataset is very large and loaded lazily, there are some tricks which makes it easier to visualizie the signal.
Using ``s.plot()`` on a lazy signal makes the library calculate a navigation image, which can be time consuming.
See `HyperSpy's big data documentation <http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html#navigator-plot>`_ for more info.
Some various ways of avoiding this issue:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.dummy_data.get_holz_simple_test_signal(lazy=True)

Using the navigator slider:

.. code-block:: python

    >>> s.plot(navigator='slider')

Using another signal as navigator, generated using :py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.virtual_annular_dark_field` or :py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.virtual_bright_field`:

.. code-block:: python

    >>> s_adf = s.virtual_annular_dark_field(25, 25, 5, 20, show_progressbar=False)
    >>> s.plot(navigator=s_adf)
    >>> s_bf = s.virtual_bright_field(25, 25, 5, show_progressbar=False)
    >>> s.plot(navigator=s_bf)


Center of mass
--------------

:py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.center_of_mass`

.. code-block:: python

    >>> s_com = s.center_of_mass(threshold=2, show_progressbar=False)
    >>> s_com.plot()


Radial integration
------------------

:py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.radial_integration`

.. code-block:: python

    >>> s.axes_manager.signal_axes[0].offset = -25
    >>> s.axes_manager.signal_axes[1].offset = -25
    >>> s_r = s.radial_integration(show_progressbar=False)
    >>> s_r.plot()


Rotating the diffraction pattern
--------------------------------

:py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.rotate_diffraction`

.. code-block:: python

    >>> s = fp.dummy_data.get_holz_simple_test_signal()
    >>> s_rot = s.rotate_diffraction(30, show_progressbar=False)
    >>> s_rot.plot()


Shifting the diffraction pattern
--------------------------------

:py:meth:`~fpd_data_processing.pixelated_stem_class.PixelatedSTEM.shift_diffraction`

.. code-block:: python

    >>> s = fp.dummy_data.get_disk_shift_simple_test_signal()
    >>> s_com = s.center_of_mass(threshold=3., show_progressbar=False)
    >>> s_com -= 25 # To shift the centre spot to (25, 25)
    >>> s_shift = s.shift_diffraction(
    ...     shift_x=s_com.inav[0].data, shift_y=s_com.inav[1].data, show_progressbar=False)
    >>> s_shift.plot()
