.. _loading_data:

============
Loading data
============

Data is loaded by using two specialized loading functions:

- One for the pixelated STEM datasets which has two spatial probe dimensions
  and two reciprocal detector dimensions: :py:func:`fpd_data_processing.io_tools.load_fpd_signal`.
- And another one for loading disk shift datasets, which consist of one navigation
  dimensions, and (normally) two signal dimensions:
  :py:func:`fpd_data_processing.io_tools.load_dpc_signal`

Both these functions are accessible through :py:mod:`fpd_data_processing.api`.

Pixelated STEM
--------------

You can either use generated test dataset, or your own data.
To generate a test pixelated STEM dataset:

.. code-block:: python

    >>> import fpd_data_processing.make_diffraction_test_data as mdtd
    >>> s = mdtd.get_holz_simple_test_signal()
    >>> s.save("test_data.hdf5")

Load the data using :py:func:`~fpd_data_processing.io_tools.load_fpd_signal`:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.load_fpd_signal("test_data.hdf5", lazy=True)

Here `lazy=True` was used.
Essentially this does not load all the data into memory, meaning very large datasets
can be processed.
However, this also means that the processing will take longer to do, and some care needs to be taken
when actually doing the processing.
See the `HyperSpy documentation <http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html>`_
for information on how to do this.

Here the dataset is fairly small, so it should easily fit into memory.:

.. code-block:: python

    >>> import fpd_data_processing.api as fp
    >>> s = fp.load_fpd_signal("test_data.hdf5")

Note: for larger datasets this might take a long time, and might use all of your memory.
This might cause the computer to crash, so be careful when loading large datasets.

To visualize the data, use plot:

.. code-block:: python

    >>> s.plot()
