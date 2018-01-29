.. _misc_functions:

=======================
Miscellaneous functions
=======================

Removing dead pixels
--------------------

Removing dead pixels in a single image: :py:func:`~fpd_data_processing.pixelated_stem_tools.remove_dead_pixels`:

.. code-block:: python

    >>> import numpy as np
    >>> import fpd_data_processing.api as fp
    >>> from fpd_data_processing.pixelated_stem_tools import remove_dead_pixels
    >>> s = fp.dummy_data.get_dead_pixel_signal()
    >>> dead_pixel_list = np.where(s.data == 0)
    >>> remove_dead_pixels(s.data, dead_pixel_list)

