.. _loading_data:

============
Loading data
============

Data is loaded by using two specialized loading functions:

- One for the :py:class:`~pixstem.pixelated_stem_class.PixelatedSTEM` datasets which has two spatial probe dimensions
  and two reciprocal detector dimensions: :py:func:`pixstem.io_tools.load_ps_signal`.
- And another one for loading disk shift datasets, which consist of one navigation
  dimensions, and (normally) two signal dimensions:
  :py:func:`pixstem.io_tools.load_dpc_signal`

Both these functions are accessible through :py:mod:`pixstem.api`.

Pixelated STEM
--------------

You can either use generated test dataset, or your own data.
To generate a test pixelated STEM dataset:

.. code-block:: python

    >>> import pixstem.dummy_data as dd
    >>> s = dd.get_holz_simple_test_signal()
    >>> s.save("test_data.hdf5")

Load the data using :py:func:`~pixstem.io_tools.load_ps_signal`:

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.load_ps_signal("test_data.hdf5", lazy=True)

Here `lazy=True` was used.
Essentially this does not load all the data into memory, meaning very large datasets
can be processed.
However, this also means that the processing will take longer to do, and some care needs to be taken
when actually doing the processing.
See the `HyperSpy documentation <http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html>`_
for information on how to do this.

Here the dataset is fairly small, so it should easily fit into memory.:

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.load_ps_signal("test_data.hdf5")

Note: for larger datasets this might take a long time, and might use all of your memory.
This might cause the computer to crash, so be careful when loading large datasets.

To visualize the data, use plot:

.. code-block:: python

    >>> s.plot()


From NumPy array to PixelatedSTEM object
****************************************

The :py:class:`~pixstem.pixelated_stem_class.PixelatedSTEM` class can also be created using a NumPy array directly.

.. code-block:: python

    >>> import pixstem.api as ps
    >>> import numpy as np
    >>> data = np.random.random((10, 15, 30, 35))
    >>> s = ps.PixelatedSTEM(data)
    >>> s
    <PixelatedSTEM, title: , dimensions: (15, 10|35, 30)>

Note that dimension 0/1 and 2/3 is flipped in the PixelatedSTEM signal, and the NumPy array.
This is due to how HyperSpy handles the input data.
In this case it leads to the signal x-dimension having a size of 35, and a y-dimension a size of 30.
While the navigation x-dimension has a size of 15, and a y-size of 10.


From Dask array to LazyPixelatedSTEM object
*******************************************

When working with very large datasets, lazy loading is preferred.
One way of doing this is by using the `dask library <https://dask.pydata.org/en/latest/>`__.
See the `HyperSpy big data documentation <http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html#working-with-big-data>`__ for more information on how to utilize lazy loading the pixstem library.

.. code-block:: python

    >>> import pixstem.api as ps
    >>> import dask.array as da
    >>> data = da.random.random((10, 7, 15, 32), chunks=((2, 2, 2, 2)))
    >>> s = ps.LazyPixelatedSTEM(data)
    >>> s
    <LazyPixelatedSTEM, title: , dimensions: (7, 10|32, 15)>


From HyperSpy signal to PixelatedSTEM
*************************************

To retain the axes manager and metadata, use the :py:func:`pixstem.io_tools.signal_to_pixelated_stem` function.

.. code-block:: python

    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> data = np.random.random((10, 15, 30, 35))
    >>> s = hs.signals.Signal2D(data)
    >>> import pixstem.io_tools as it
    >>> s_new = it.signal_to_pixelated_stem(s)


.. _load_dpc_data:

Differential phase contrast (beam shift) data
---------------------------------------------

Differential phase contrast (DPC) datasets are loaded using :py:func:`pixstem.io_tools.load_dpc_signal`.
These datasets must have one navigation dimensions with two indices, where the first navigation index is the x-direction beam shift, and the second navigation dimension is the y-direction beam shift.
The signal dimensions must be either two, one or zero, giving either :py:class:`~pixstem.pixelated_stem_class.DPCSignal2D`, :py:class:`~pixstem.pixelated_stem_class.DPCSignal1D` or :py:class:`~pixstem.pixelated_stem_class.DPCBaseSignal`.

Files saved using HyperSpy can also be opened directly, as long as the dataset has one navigation dimension with a shape of 2.

You can either use generated test dataset, or your own data.
To generate a test DPC dataset:

.. code-block:: python

    >>> import pixstem.dummy_data as dd
    >>> s = dd.get_simple_dpc_signal()
    >>> s.save("test_dpc_data.hdf5")

To load the test file (or your own file):

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.load_dpc_signal("test_dpc_data.hdf5")

Plotting the data:

.. code-block:: python

    >>> s.plot()
    >>> s.get_color_signal().plot()


From NumPy array to DPCSignal objects
*************************************


The :py:class:`~pixstem.pixelated_stem_class.DPCSignal2D` object can be created using



.. code-block:: python

    >>> import pixstem.api as ps
    >>> import numpy as np
    >>> data = np.random.random((2, 21, 54))
    >>> s = ps.DPCSignal2D(data)
    >>> s
    <DPCSignal2D, title: , dimensions: (2|54, 21)>


Note the switch of the x/y signal axis.

The :py:class:`~pixstem.pixelated_stem_class.DPCSignal1D` object can be created using:

.. code-block:: python

    >>> data = np.random.random((2, 109))
    >>> s = ps.DPCSignal1D(data)
    >>> s
    <DPCSignal1D, title: , dimensions: (2|109)>


The :py:class:`~pixstem.pixelated_stem_class.DPCBaseSignal` object can be created using:

.. code-block:: python

    >>> data = np.random.random((2, ))
    >>> s = ps.DPCBaseSignal(data)
    >>> s
    <DPCBaseSignal, title: , dimensions: (|2)>
