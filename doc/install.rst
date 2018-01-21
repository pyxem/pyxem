.. _install:

==========
Installing
==========

Installing in Windows
---------------------

WinPython HyperSpy installer
****************************

Currently, the easiest way to install is using the WinPython HyperSpy installer.
Firstly download and install the `WinPython HyperSpy bundle <http://hyperspy.org/download.html#windows-bundle-installers>`_:
HyperSpy-1.3 for Windows 64-bits.

After installing the bundle, there should be a folder in the start menu called "HyperSpy WinPython Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install fpd_data_processing

To check everything is working correctly, go to the "HyperSpy WinPython Bundle" and start "Jupyter QtConsole".
This will open a new window. In this window, run:

.. code-block:: python

    %matplotlib qt4
    import fpd_data_processing.api as fp

If this works, continue with the :ref:`using_pixelated_stem_class`.


Installing in Linux
-------------------

The recommended way to install is using PIP, which is a package manager for python.
It is recommended to first install the precompiled dependencies using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included as fpd_data_processing relies heavily on the modelling and visualization functionality in HyperSpy.

Ubuntu 17.10
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt4 python3-lxml
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ pip3 install --upgrade pip
    $ pip3 install --user fpd_data_processing


Ubuntu 16.04
************

Due to the old version of matplotlib in the repository, the system matplotlib has to be removed.
This is due to conflicts between the newer 2.x version in PIP and the older 1.5.x version in the repository.

In addition, due to a recent bug with HyperSpy and matplotlib 2.1.x, matplotlib 2.0.2 has to be installed.

.. code-block:: bash

    $ sudo apt-get install python3-pip python3-numpy python3-scipy python3-h5py ipython3 python3-natsort python3-sklearn python3-dill python3-ipython-genutils python3-pyqt4
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ sudo apt-get remove python3-matplotlib
    $ pip3 install --user --upgrade pip
    $ pip3 install --user matplotlib==2.0.2
    $ pip3 install --user fpd_data_processing


Starting fpd_data_processing
****************************

To check that everything is working, open a terminal and run :code:`ipython3 --matplotlib qt4`. In the ipython terminal run:

.. code-block:: python

    import fpd_data_processing.api as fp

If this works, continue with the :ref:`using_pixelated_stem_class`.
If you get some kind of error, please report it as a New issue on the `fpd_data_processing GitLab <https://gitlab.com/fast_pixelated_detectors/fpd_data_processing/issues>`_.
Note, having the system and pip version of matplotlib installed at the same might cause an error with matplotlib not finding matplotlib.external.
The easiest way of fixing this is by removing the system version of matplotlib.


Development version
-------------------

Grab the development version using the version control system git:

.. code-block:: bash

    $ git clone git@gitlab.com:fast_pixelated_detectors/fpd_data_processing.git

Then install it using pip:

.. code-block:: bash

    $ cd fpd_data_processing
    $ pip3 install -e .
