.. _LoadingData:

############
Loading Data
############

Data can be loaded from many different 4-D STEM detectors using the package
`rosettasciio<https://hyperspy.org/rosettasciio>`_. The file types currently supported are listed
`here<https://hyperspy.org/rosettasciio/supported_formats/index.html>`_.

We are always looking to add more detectors to the list. If you have a detector that you
would like to see added, please raise an `issue<https://github.com/hyperspy/rosettasciio/issues>`_
on the rosettasciio GitHub page and we will do our best to work with you to add it.

We are also looking to add support for faster loading of data from detectors that we already
support.  This means adding support for distributed file loading, which allows for operating
on data using multiple computers connected over a network.

To load data from a file, we use the :func:`~hyperspy.api.load` function.  This function
automatically detects the file type and returns a hyperspy signal object.  For example,

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> s = hs.load("data/4DSTEM_simulation.hspy")
    >>> s # Signal2D object

If we want to load the signal as a specific signal type,we can use the ``signal_type`` argument.
For example,

.. code-block:: python

    >>> s = hs.load("data/4DSTEM_simulation.hspy", signal_type="electron_diffraction"
    >>> s # ElectronDiffraction2D object (defined in pyxem!)

Or we can cast the signal to a specific signal type after loading it.  For example,

.. code-block:: python

    >>> s = hs.load("data/4DSTEM_simulation.hspy")
    >>> s = s.set_signal_type("electron_diffraction")
    >>> s # ElectronDiffraction2D object (defined in pyxem!)

Note that when we save the signal to a ``.zspy`` file or a ``.hspy`` file, the signal type
is saved as metadata.  This means that when we load the signal, we no longer need to specify
the signal type.  For example,

.. code-block:: python

    >>> s = hs.load("data/4DSTEM_simulation.zspy")
    >>> s.set_signal_type("electron_diffraction")
    >>> s.save("data/4DSTEM_simulation_2.zspy")
    >>> s = hs.load("data/4DSTEM_simulation_2.zspy")
    >>> s # ElectronDiffraction2D object (defined in pyxem!)