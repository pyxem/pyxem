.. _install:

==========
Installing
==========

Installing in Windows
---------------------

Anaconda Python environment
***************************

Currently, the easiest way to install pixStem is using the Anaconda python environment `Anaconda environment <https://www.continuum.io/downloads>`_,
Install HyperSpy, then pixStem via the `Anaconda prompt` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ conda install hyperspy -c conda-forge
    $ pip install hyperspy_gui_traitsui
    $ pip install pixstem

To check everything is working correctly, go to "Anaconda3" in the start menu, and start "Jupyter Notebook".
This will open a browser window (or a new browser tab).
Start a new Python 3 notebook, and run in the first cell:

.. code-block:: python

    %matplotlib qt5
    import pixstem.api as ps

If this works, continue with the :ref:`using_pixelated_stem_class`.


WinPython HyperSpy installer
****************************

Alternatively, the WinPython HyperSpy bundle can be used.
Firstly download and install the `WinPython HyperSpy bundle <://github.com/hyperspy/hyperspy-bundle/releases>`_:
HyperSpy-1.3 for Windows 64-bits (get the most recent version).

After installing the bundle, there should be a folder in the start menu called "HyperSpy WinPython Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install pixstem

To check everything is working correctly, go to the "HyperSpy WinPython Bundle" and start "Jupyter QtConsole".
This will open a new window. In this window, run:

.. code-block:: python

    %matplotlib qt5
    import pixstem.api as ps

If this works, continue with the :ref:`using_pixelated_stem_class`.


Installing in Linux
-------------------

The recommended way to install is using PIP, which is a package manager for python.
It is recommended to first install the precompiled dependencies using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included as pixStem relies heavily on the modelling and visualization functionality in HyperSpy.

Ubuntu 17.10
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt5 python3-lxml
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ pip3 install --upgrade pip
    $ pip3 install --user pixstem


Ubuntu 16.04
************

Due to the old version of matplotlib in the repository, the system matplotlib has to be removed.
This is due to conflicts between the newer 2.x version in PIP and the older 1.5.x version in the repository.

In addition, due to a recent bug with HyperSpy and matplotlib 2.1.x, matplotlib 2.0.2 has to be installed.

.. code-block:: bash

    $ sudo apt-get install python3-pip python3-numpy python3-scipy python3-h5py ipython3 python3-natsort python3-sklearn python3-dill python3-ipython-genutils python3-pyqt5
    $ sudo apt-get install python3-sympy --no-install-recommends
    $ sudo apt-get remove python3-matplotlib
    $ pip3 install --user --upgrade pip
    $ pip3 install --user matplotlib==2.0.2
    $ pip3 install --user pixstem


Starting pixStem
****************

To check that everything is working, open a terminal and run :code:`ipython3 --matplotlib qt5`. In the ipython terminal run:

.. code-block:: python

    import pixstem.api as ps

If this works, continue with the :ref:`using_pixelated_stem_class`.
If you get some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


Development version
-------------------

Grab the development version using the version control system git:

.. code-block:: bash

    $ git clone git@gitlab.com:pixstem/pixstem.git

Then install it using pip:

.. code-block:: bash

    $ cd pixstem
    $ pip3 install -e .
