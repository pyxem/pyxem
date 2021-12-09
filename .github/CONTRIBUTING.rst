Contributor Guide
=================

This guide is intended to get new developers started with contributing to pyxem.

Many potential contributors will be scientists with much expert knowledge but
potentially little experience with open-source code development. This guide is
primarily aimed at this audience, helping to reduce the barrier to contribution.


Start using pyxem
-----------------

The best way to start understanding how pyxem is to use it.

For developing the code the home of pyxem is on github and you'll see that
a lot of this guide boils down to using that platform well. so visit the
following link and poke around the code, issues, and pull requests: `pyxem
on Github <https://github.com/pyxem/pyxem>`_.

It's probably also worth visiting the `Github <https://github.com/>`_ home page
and going through the "boot camp" to get a feel for the terminology.

In brief, to give you a hint on the terminology to search for, the contribution
pattern is:

1. Setup git/github if you don't have it.
2. Fork pyxem on github.
3. Checkout your fork on your local machine.
4. Create a new branch locally where you will make your changes.
5. Push the local changes to your own github fork.
6. Create a pull request (PR) to the official pyxem repository.

Note: You cannot mess up the main pyxem project. So when you're starting out be
confident to play, get it wrong, and if it all goes wrong you can always get a
fresh install of pyxem!

PS: If you choose to develop in Windows/Mac you may find `Github Desktop
<https://desktop.github.com>`_ useful.


Questions?
----------

Open source projects are all about community - we put in much effort to make
good tools available to all and most people are happy to help others start out.
Everyone had to start at some point and the philosophy of these projects
centers around the fact that we can do better by working together.

Much of the conversation happens in 'public' using the 'issues' pages on
`Github <https://github.com/pyxem/pyxem/issues>`_ -- doing things in public can
be scary but it ensures that issues are identified and logged until dealt with.
This is also a good place to make a proposal for some new feature or tool that
you want to work on.


Good coding practice
====================

The most important aspects of good coding practice are: (1) to work in managable
branches, (2) develop good code style, (3) write tests for new functions, and (4)
document what the code does. Tips on these points are provided below.


Use git to work in manageable branches
--------------------------------------

Git is an open source "version control" system that enables you to can separate
out your modifications to the code into many versions (called branches) and
switch between them easily. Later you can choose which version you want to have
integrated into pyXem.

You can learn all about Git `here <http://www.git-scm.com/about>`_!

The most important thing to separate your contributions so that each branch is
small advancement on the "master" code or on another branch.


Get the style right
-------------------

pyxem follows the Style Guide for Python Code - these are just some rules for
consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_.

Consistent code formatting in pyxem is achieved by using the `Black
<https://black.readthedocs.io/en/stable/>`_ code formatter, which can be
installed using conda:

.. code:: bash

   conda install black

The code can then be formatted correctly by running black from the pyxem project
folder:

.. code:: bash

   black pyxem


Writing tests
-------------

pyxem aims to have all of the functions within it tested, which involves writing
short methods that call the functions and check output values against known
answers. Good tests should depend on as few other features as possible so that
when they break we know exactly what caused it.

pyxem uses the `pytest <http://doc.pytest.org/>`_ library for testing, which can
be installed using conda:

.. code:: bash

   conda install pyxem

The tests reside in the ``pyxem.tests`` module and can be run locally from the
pyxem project folder:

.. code:: bash

   pytest pyxem


Useful hints on testing:

* When comparing integers, it's fine to use ``==``. When comparing floats use
  something like assert ``np.allclose(shifts, shifts_expected, atol=0.2)``
* ``@pytest.mark.parametrize()`` is a very convenient decorator to test several
  parameters of the same function without having to write to much repetitive
  code, which is often error-prone. See `pytest documentation for more details
  <http://doc.pytest.org/en/latest/parametrize.html>`_.
* We test the code coverage on pull requests, you can check the coverage on a
  local branch using

.. code:: bash

   pytest --cov=pyxem

* Some useful fixtures (e.g. a basic diffraction pattern) can be found in
  conftest.py, you can just call these directly in the test suite.


Write documentation
-------------------

Docstrings -- written at the start of a function and give essential information
about how it should be used, such as which arguments can be passed to it and
what the syntax should be. The docstrings need to follow the `numpy specification
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_,
as shown in `this example <https://github.com/numpy/numpy/blob/master/doc/example.py>`_.


Learn more
==========

1. HyperSpy's `contribution guide <https://hyperspy.readthedocs.io/en/latest/dev_guide/index.html>`__: a lot of nice information on how to contribute to a scientific Python project.
2. The Python programming language, `for beginners <https://www.python.org/about/gettingstarted/>`__.
