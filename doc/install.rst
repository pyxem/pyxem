.. _install:

==========
Installing
==========

.. _install_windows:

Installing in Windows
---------------------

Anaconda Python environment
***************************

Currently, the easiest way to install pixStem is using the Anaconda python environment `Anaconda environment <https://www.continuum.io/downloads>`_,
Install HyperSpy, then pixStem via the ``Anaconda prompt`` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ conda install hyperspy -c conda-forge
    $ pip install pixstem

If everything installed, continue to :ref:`starting pixStem in Windows <start_pixstem_windows>`.
If you got some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


WinPython HyperSpy installer
****************************

Alternatively, the WinPython HyperSpy bundle can be used.
Firstly download and install the `WinPython HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle/releases>`_:

After installing the bundle, there should be a folder in the start menu called "HyperSpy Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install pixstem

If everything installed, continue to :ref:`starting pixStem in Windows <start_pixstem_windows>`.
If you got some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


Installing in MacOS
-------------------

Install the Anaconda python environment: `Anaconda environment <https://www.continuum.io/downloads>`_, and through the ``Anaconda prompt`` install HyperSpy and pixStem:

.. code-block:: bash

    $ conda install hyperspy -c conda-forge
    $ pip install pixstem

If everything installed, continue to :ref:`starting pixStem in MacOS <start_pixstem_macos>`.
If you got some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


Installing in Linux
-------------------

The recommended way to install pixStem is using PIP, which is a package manager for python.
It is recommended to first install the precompiled dependencies using the system package manager.

`HyperSpy <http://hyperspy.org/>`_ is also included as pixStem relies heavily on the modelling and visualization functionality in HyperSpy.

Ubuntu 18.04
************

.. code-block:: bash

    $ sudo apt-get install ipython3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-sklearn python3-skimage python3-h5py python3-dask python3-traits python3-tqdm python3-pint python3-dask python3-pyqt5 python3-lxml python3-sympy python3-sparse python3-statsmodels python3-numexpr python3-ipykernel python3-jupyter-client python3-requests python3-dill python3-natsort
    $ pip3 install --user pixstem

If everything installed, continue to :ref:`starting pixStem in Linux <start_pixstem_linux>`.
If you got some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


Development version
-------------------

Grab the development version using the version control system git:

.. code-block:: bash

    $ git clone https://gitlab.com/pixstem/pixstem.git

Then install it using pip:

.. code-block:: bash

    $ cd pixstem
    $ pip3 install -e .
