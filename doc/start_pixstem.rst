.. _start_pixstem:


=============
Start pixStem
=============

Starting Python
---------------

The first step is starting an interactive Jupyter Notebook environment.

.. _start_pixstem_linux:

Linux
^^^^^

Open a terminal and start ``ipython3``:

.. code-block:: bash

    $ ipython3 notebook


If ``ipython3`` is not available, try ``ipython``:

.. code-block:: bash

    $ ipython notebook


This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import pixstem.api as ps

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


.. _start_pixstem_windows:

Windows
^^^^^^^

This depends on the installation method:

* If the HyperSpy bundle was installed, go to the "HyperSpy Bundle" in the start-menu and start "Jupyter Notebook".
* If Anaconda was used, go to "Anaconda3" in the start menu, and start "Jupyter Notebook".

This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import pixstem.api as ps

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


.. _start_pixstem_macos:

MacOS
^^^^^

Open the Terminal, and write:

.. code-block:: bash

    $ jupyter notebook


This will open a browser window (or a new browser tab).
Press the "New" button (top right), and start a Python 3 Notebook.
In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the Jupyter Notebook guide <https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_.

.. code-block:: python

    %matplotlib nbagg
    import pixstem.api as ps

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `pixStem GitLab <https://gitlab.com/pixstem/pixstem/issues>`_.


.. _tutorials:

Tutorials
---------

To get you started on using pixStem there are tutorials available.
The first tutorial :ref:`loading_data` shows how to load the data, while :ref:`using_pixelated_stem_class` shows how to use the ``PixelatedSTEM`` class, which contains most of pixStem's functionality.

The `>>>` used in the tutorials and documentation means the comment should be typed inside some kind of Python prompt, and can be copy-pasted directly into the *Jupyter Notebooks*.


pixStem demos
^^^^^^^^^^^^

In addition to the guides on this webpage, another good resource is the `Introduction to pixStem notebook <https://gitlab.com/pixstem/pixstem_demos/blob/release/introduction_to_pixstem.ipynb>`_.
This is a pre-filled Jupyter Notebooks showing various aspects of pixStem's functionality.
