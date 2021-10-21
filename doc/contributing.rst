=============
Contributing
=============

pyxem is meant to be a community maintained project. We welcome
contributions in the form of bug reports, documentation, code, feature requests,
and more. These guidelines provide resources on how best to contribute.

For new users, checking out the `GitHub guides <https://guides.github.com>`_ are
recommended.

Questions, comments, and feedback
=================================

Have any questions, comments, suggestions for improvements, or any other
inquiries regarding the project? Feel free to
`ask a question <https://github.com/pyxem/pyxem/discussions>`_,
`open an issue <https://github.com/pyxem/pyxem/issues>`_ or
`make a pull request <https://github.com/pyxem/pyxem/pulls>`_ in our GitHub
repository.


.. _setting-up-a-development-installation:

First Time Contributing
=======================
One of the best things that first time contributors can do is help to improve our documentation, our
`demos https://github.com/pyxem/pyxem-demos`_ or report a bug.

If you have a workflow that you don't see covered in the demos feel free to submit an `issue
<https://github.com/pyxem/pyxem-demos/issues>`_ and someone can help you with the process of adding
to the demos.


Setting up a development installation
=====================================

You need a `fork <https://guides.github.com/activities/forking/#fork>`_ of the
`repository <https://github.com/pyxem/pyxem>`_ in order to make changes
to pyxem.

Make a local copy of your forked repository and change directories::

    $ git clone https://github.com/your-username/pyxem.git
    $ cd pyxem

Set the ``upstream`` remote to the main pyxem repository::

    $ git remote add upstream https://github.com/pyxem/pyxem.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution
<https://docs.conda.io/en/latest/miniconda.html>`_::

   $ conda create --name pyxem
   $ conda activate pyxem

Then, install the required dependencies while making the development version
available globally (in the ``conda`` environment)::

   $ pip install --editable .[dev]

This installs all necessary development dependencies, including those for
running tests and building documentation.

Code style
==========

The code making up pyxem is formatted closely following the `Style Guide for
Python Code <https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code
style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_. We
use `pre-commit <https://pre-commit.com>`_ to run ``black`` automatically prior
to each local commit. Please install it in your environment::

    $ pre-commit install

Next time you commit some code, your code will be formatted inplace according
to our `black configuration
<https://github.com/pyxem/pyxem/blob/master/pyproject.toml>`_.

Note that ``black`` won't format `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_.

Package imports should be structured into three blocks with blank lines between
them (descending order): standard library (like ``os`` and ``typing``), third
party packages (like ``numpy`` and ``hyperspy``) and finally pyxem imports.

Making changes
==============

Create a new feature branch::

    $ git checkout master -b your-awesome-feature-name

When you've made some changes you can view them with::

    $ git status

Add and commit your created, modified or deleted files::

   $ git add my-file-or-directory
   $ git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered
email <https://github.com/settings/emails>`_ as the author. You can set this up
following `this GitHub guide
<https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Keeping your branch up-to-date
==============================

Switch to the ``master`` branch::

   $ git checkout master

Fetch changes and update ``master``::

   $ git pull upstream master --tags

Update your feature branch::

   $ git checkout your-awesome-feature-name
   $ git merge master

Sharing your changes
====================

Update your remote branch::

   $ git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to
pyxem's ``master`` branch. Good job!

Building and writing documentation
==================================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting
functionality. Install necessary dependencies to build the documentation::

   $ pip install --editable .[doc]

Then, build the documentation from the ``doc`` directory::

   $ cd doc
   $ make html

The documentation's HTML pages are built in the ``doc/build/html`` directory
from files in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
plaintext markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/pyxem/doc/build/html/index.html`` in the
address bar.

Tips for writing Jupyter Notebooks that are meant to be converted to reST text
files by `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- Use ``_ = ax[0].imshow(...)`` to disable Matplotlib output if a Matplotlib
  command is the last line in a cell.
- Refer to our API reference with this general MD
  ``[azimuthal_integrator2d()](reference.rst#pyxem.signals.DiffractionSignal2D.azimuthal_integrator2d)``. Remember
  to add the parentheses ``()``.
- Reference external APIs via standard MD like
  ``[Signal2D](http://hyperspy.org/hyperspy-doc/current/api/hyperspy._signals.signal2d.html)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook
  must be added to the gallery in the README.rst to be included in the
  documentation pages.

Running and writing tests
=========================

All functionality in pyxem is tested via the `pytest
<https://docs.pytest.org>`_ framework. The tests reside in a ``test`` directory
within each module. Tests are short methods that call functions in pyxem
and compare resulting output values with known answers. Install necessary
dependencies to run the tests::

   $ pip install --editable .[tests]


To run the tests::

   $ pytest --cov --pyargs pyxem

The ``--cov`` flag makes `coverage.py
<https://coverage.readthedocs.io/en/latest/>`_ print a nice report in the
terminal. For an even nicer presentation, you can use ``coverage.py`` directly::

   $ coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect
the coverage in more detail.

Docstring examples are tested
`with pytest <https://docs.pytest.org/en/stable/doctest.html>`_ as well::

   $ pytest --doctest-modules --ignore-glob=pyxem/*/tests

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/pyxem/actions>`_ to ensure
that pyxem can be installed on Windows, macOS and Linux (Ubuntu). After a
successful installation, the CI server runs the tests. After the tests return no
errors, code coverage is reported to `Coveralls
<https://coveralls.io/github/pyxem/pyxem?branch=master>`_.
