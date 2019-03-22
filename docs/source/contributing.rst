Contributing
============

This guide is intended to get new developers started with contributing to pyxem.

Many potential contributors will be scientists with much expert knowledge but
potentially little experience with open-source code development. This guide is
primarily aimed at this audience, helping to reduce the barrier to contribution.


Start using pyXem
-----------------

The best way to start understanding how pyXem is to use it.

For developing the code the home of pyXem is on github and you'll see that
a lot of this guide boils down to using that platform well. so visit the
following link and poke around the code, issues, and pull requests: `pyXem
on Github <https://github.com/pyxem/pyxem>`_.

It's probably also worth visiting the `Github <https://github.com/>`_ home page
and going through the "boot camp" to get a feel for the terminology.

In brief, to give you a hint on the terminology to search for, the contribution
pattern is:

1. Setup git/github if you don't have it.
2. Fork pyXem on github.
3. Checkout your fork on your local machine.
4. Create a new branch locally where you will make your changes.
5. Push the local changes to your own github fork.
6. Create a pull request (PR) to the official pyXem repository.

Note: You cannot mess up the main pyXem project. So when you're starting out be
confident to play, get it wrong, and if it all goes wrong you can always get a
fresh install of pyXem!

PS: If you choose to develop in Windows/Mac you may find `Github Desktop
<https://desktop.github.com>`_ useful.


Good coding practice
--------------------

The most important aspects of good coding practice are: (1) to work in managable
branches, (2) develop good code style, (3) write tests for new functions, and (4)
document what the code does. Tips on these points are provided below.


Use git to work in managable branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Git is an open source "version control" system that enables you to can separate
out your modifications to the code into many versions (called branches) and
switch between them easily. Later you can choose which version you want to have
integrated into pyXem.

You can learn all about Git `here <http://www.git-scm.com/about>`_!

The most important thing to separate your contributions so that each branch is
small advancement on the "master" code or on another branch.


Get the style right
^^^^^^^^^^^^^^^^^^^

pyXem follows the Style Guide for Python Code - these are just some rules for
consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_.

To enforce this, we require that the following auto correction is applied at the
end of pull request. The simplest option is to run (from the home directory of
pyxem)

Note that if you have recently run tests locally you may have a test generated
file pyxem/file_01.pickle on your machine, which should be deleted prior to
running the following code.

Linux:

.. code:: bash

    chmod +x pepstorm.sh
    ./pepstorm.sh
    git add .
    git commit -m "autopep8 corrections"

Windows:

.. code:: batch

    pepstorm.bat
    git add .
    git commit -m "autopep8 corrections"


Writing tests
^^^^^^^^^^^^^

pyXem aims to have all of the functions within it tested, which involves writing
short methods that call the functions and check output values agains known
answers. Good tests should depend on as few other features as possible so that
when they break we know exactly what caused it.

pyXem uses the `pytest <http://doc.pytest.org/>`_ library for testing. The
tests reside in the ``pyxem.tests`` module. To run them (from the pyXem project
folder):

.. code:: bash

   pytest


Useful hints on testing:

* When comparing integers, it's fine to use ``==``. When comparing floats use something like assert ``np.allclose(shifts,shifts_expected,atol=0.2)``
* ``@pytest.mark.parametrize()`` is a very convenient decorator to test several
  parameters of the same function without having to write to much repetitive
  code, which is often error-prone. See `pytest documentation for more details
  <http://doc.pytest.org/en/latest/parametrize.html>`_.
* We test the code coverage on pull requests, you can check the coverage on a
  local branch using

.. code:: bash

   pytest --cov=pyxem

* Some useful fixtures (a basic diffraction pattern, a basic structure...) can
  be found in conftest.py, you can just call these directly in the test suite.


Write documentation
^^^^^^^^^^^^^^^^^^^

Docstrings -- written at the start of a function and give essential information
about how it should be used, such as which arguments can be passed to it and
what the syntax should be. The docstrings need to follow the `numpy specification
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_,
as shown in `this example <https://github.com/numpy/numpy/blob/master/doc/example.py>`_.


Learn more
----------

1. HyperSpy's `contribution guide <http://hyperspy.org/hyperspy-doc/current/dev_guide.html#developer-guide>`__: a lot of nice information on how to contribute to a scientific Python project.
2. The Python programming language, `for beginners <https://www.python.org/about/gettingstarted/>`__.
