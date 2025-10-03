Big Data
---------

4-D STEM datasets are large and difficult to work with.  In ``pyxem`` we try to get around this by
using a lazy loading approach.  This means that the data is not loaded into memory until it is
needed.

This is a very powerful approach, but it can be confusing at first. In this guide we will
discuss how to work with large datasets in ``pyxem``.

.. note::
    If you want more information on working with large datasets in ``pyxem``, please see the
    `Big Data <https://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html>_` section of
    the `HyperSpy User Guide <https://hyperspy.org/hyperspy-doc/current/user_guide/index.html>_`.

Loading and Plotting a Dataset
------------------------------

Let's start by loading a dataset.  We will use the ``load`` function to load a dataset from
HyperSpy

.. code-block::

    import hyperspy.api as hs
    s = hs.load("big_data.zspy", lazy=True)
    s

The dataset here is not loaded into memory here, so this should happen instantaneously. We can
then plot the dataset.

.. code-block::

    s.plot()

Which (in general) will be very slow.  This is because entire dataset is loaded into memory (chuck by chunk)
to create a navigator image. In many cases a HAADF dataset will be collected as well as a 4-D STEM dataset.
In this case we can set the navigator to the HAADF dataset instead.

.. code-block:: python

    haadf = hs.load("haadf.zspy") # load the HAADF dataset
    s.plot(navigator=haadf) # happens instantaneously

This is much faster as the navigator doesn't need to be computed and instead only 1 chunk needs to
be loaded into memory before plotting!

You can also set the navigator so that by default it is used when plotting.

.. code-block:: python

    haadf = hs.load("haadf.zspy") # load the HAADF dataset
    s.navigator = haadf
    s.plot()


Distributed Computing
---------------------

In ``pyxem`` we can use distributed computing to speed up the processing of large datasets.  This
is done using the `Dask <https://dask.org/>`_ library.  Dask is a library for parallel computing
in Python.  It is very powerful and can be used to speed up many different types of computations.

The first step is to set up a `Dask Client <https://distributed.dask.org/en/latest/client.html>`_.
This can be done using the distributed scheduler.

.. code-block::

    from dask.distributed import Client
    client = Client()

This will start a local cluster on your machine.

If you want to use a remote cluster using a scheduler such as `Slurm <https://slurm.schedmd.com/>`_
you can do so by using the `dask-jobqueue <https://jobqueue.dask.org/en/latest/>`_ library.
This is a library that allows you to use a scheduler to start a cluster on a remote machine.

.. code-block::

    from dask_jobqueue import SLURMCluster
    cluster = SLURMCluster()
    client = Client(cluster)

