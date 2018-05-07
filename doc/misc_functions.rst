.. _misc_functions:

=======================
Miscellaneous functions
=======================

Removing dead pixels
--------------------

Removing dead pixels in PixelatedSTEM signals :py:func:`~fpd_data_processing.pixelated_stem_tools.find_and_remove_dead_pixels`:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> import fpd_data_processing.pixelated_stem_tools as pst
    >>> s = fp.dummy_data.get_dead_pixel_signal()
    >>> pst.find_and_remove_dead_pixels(s)

