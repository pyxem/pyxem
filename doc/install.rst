.. _install:

==========
Installing
==========

Install using PyPi:

.. code-block:: bash

    $ pip3 install fpd_data_processing hyperspy_gui_ipywidgets hyperspy_gui_traitsui

Continue with the tutorial: :ref:`using_pixelated_stem_class`.


For Ubuntu 16.04
----------------

Note: the matplotlib package in Ubuntu 16.04 is too old, which causes conflicts when installing the newer version through PyPI.

.. code-block:: bash

    $ sudo apt-get remove -qy python3-matplotlib

