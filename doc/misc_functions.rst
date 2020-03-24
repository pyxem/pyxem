.. _misc_functions:

=======================
Miscellaneous functions
=======================

Removing dead pixels
--------------------

Removing dead pixels in PixelatedSTEM signals :py:func:`~pixstem.pixelated_stem_tools.find_and_remove_dead_pixels`:

.. code-block:: python

    >>> import pixstem.api as ps
    >>> import pixstem.pixelated_stem_tools as pst
    >>> s = ps.dummy_data.get_dead_pixel_signal()
    >>> pst.find_and_remove_dead_pixels(s)

