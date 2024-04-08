.. _LoadingData:

############
Loading Data
############

Data can be loaded from many different 4-D STEM detectors using the package
`rosettasciio<https://hyperspy.org/rosettasciio>_`. The file types currently supported are listed
`here<https://hyperspy.org/rosettasciio/supported_formats/index.html>_`.

We are always looking to add more support for more file formats and detectors.
If you have a files or detectors that you would like to see supported,
please raise an `issue<https://github.com/hyperspy/rosettasciio/issues>_`
on the rosettasciio GitHub page and we will do our best to work with you to add it.

We are also looking to add support for faster loading of data from detectors that we already
support.  This means adding support for distributed file loading, which allows for operating
on data using multiple computers connected over a network.

To load data from a file, we use the :func:`hyperspy.api.load` function.  The
:func:`hyperspy.api.load` function will return different type of object depending what
file is being loaded. For example, it can return HyperSpy generic signal object
(e. g. :class:`hyperspy.api.signals.Signal2D`) or domain specific signal object
(e. g. :class:`pyxem.signals.ElectronDiffraction2D`).

With some file formats, it is possible to assign the data to a suitable signal type when loading the file
and when possible it is done automatically. For example, loading:external+rsciio:`.blo<blockfile-format>` will
return :class:`pyxem.signals.ElectronDiffraction2D` object but in situations, where a generic signal is returned,
a domain-specific signal can be specified as follow:

.. code-block::

    s = hs.load("data/4DSTEM_simulation.hspy", signal_type="electron_diffraction")
    s # ElectronDiffraction2D object (defined in pyxem!)

Or we can cast the signal to a specific signal type after loading it.  For example,

.. code-block::

    s = hs.load("data/4DSTEM_simulation.hspy")
    s = s.set_signal_type("electron_diffraction")
    s # ElectronDiffraction2D object (defined in pyxem!)

Note that when we save the signal to a ``.zspy`` file or a ``.hspy`` file, the signal type
is saved as metadata.  This means that when we load the signal, we no longer need to specify
the signal type.  For example,

.. code-block::

    s = hs.load("data/4DSTEM_simulation.zspy")
    s.set_signal_type("electron_diffraction")
    s.save("data/4DSTEM_simulation_2.zspy")
    s = hs.load("data/4DSTEM_simulation_2.zspy")
    s # ElectronDiffraction2D object (defined in pyxem!)