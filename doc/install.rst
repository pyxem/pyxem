.. _install:

==========
Installing
==========

Download the source tar-file from the `fpd_data_processing GitLab <https://gitlab.com/pixelated_stem/fpd_data_processing/repository/master/archive.tar.gz>`__.
Then install using:

.. code-block:: bash

    $ pip3 install fpd_data_processing-master-*.tar.gz


Notes: old matplotlib
---------------------

Note: the matplotlib package in Ubuntu 16.04 is too old, which causes conflicts when installing the newer version through PyPI.

.. code-block:: bash

    $ sudo apt-get remove -qy python3-matplotlib

