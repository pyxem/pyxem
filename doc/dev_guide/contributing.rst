.. _contributing:

============
Contributing
============

pyxem is a community maintained project. We welcome contributions in the form of bug
reports, documentation, code, feature requests, and more. These guidelines provide
resources on how best to contribute.

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

One of the best things that first time contributors can do is help to improve our
documentation, our `demos <https://github.com/pyxem/pyxem-demos>`_ or report a bug.

If you have a workflow that you don't see covered in the demos feel free to submit an
`issue <https://github.com/pyxem/pyxem-demos/issues>`_ and someone can help you with the
process of adding to the demos.

Setting up a development installation
=====================================

You need a `fork <https://docs.github.com/en/get-started/quickstart/contributing-to-projects#about-forking>`_ of the
`repository <https://github.com/pyxem/pyxem>`_ in order to make changes to pyxem.

Make a local copy of your forked repository and change directories::

    git clone https://github.com/your-username/pyxem.git
    cd pyxem

Set the ``upstream`` remote to the main pyxem repository::

    git remote add upstream https://github.com/pyxem/pyxem.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`_::

   conda create --name pyxem
   conda activate pyxem

Then, install the required dependencies while making the development version available
globally (in the ``conda`` environment)::

   pip install --editable .[dev]

This installs all necessary development dependencies, including those for running tests
and building documentation.

Code style
==========

The code making up pyxem is formatted closely following the `Style Guide for Python Code
<https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code style
<https://black.readthedocs.io/en/stable/the_black_code_style/index.html>`_. Note that
``black`` won't format `docstrings <https://www.python.org/dev/peps/pep-0257/>`_.

Package imports should be structured into three blocks with blank lines between
them (descending order): standard library (like ``os`` and ``typing``), third
party packages (like ``numpy`` and ``hyperspy``) and finally pyxem imports.

Making changes
==============

Create a new feature branch::

    git checkout main -b your-awesome-feature-name

When you've made some changes you can view them with::

    git status

Add and commit your created, modified or deleted files::

   git add my-file-or-directory
   git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered email
<https://github.com/settings/emails>`_ as the author. You can set this up following
`this GitHub guide <https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Keeping your branch up-to-date
==============================

Switch to the ``main`` branch::

   git checkout main

Fetch changes and update ``main``::

   git pull upstream main --tags

Update your feature branch::

   git checkout your-awesome-feature-name
   git merge main

Sharing your changes
====================

Update your remote branch::

   git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to pyxem's
``main`` branch. Good job!



Deprecations
============
We attempt to adhere to semantic versioning as best we can. This means that as little,
ideally no, functionality should break between minor releases. Deprecation warnings
are raised whenever possible and feasible for functions/methods/properties/arguments,
so that users get a heads-up one (minor) release before something is removed or changes,
with a possible alternative to be used.

The decorator should be placed right above the object signature to be deprecated::

    @deprecate(since=0.8, removal=0.9, alternative="bar")
    def foo(self, n):
        return n + 1

    @property
    @deprecate(since=0.9, removal=0.10, alternative="another", is_function=True)
    def this_property(self):
        return 2

Running and writing tests
=========================

All functionality in pyxem is tested via the `pytest <https://docs.pytest.org>`_
framework. The tests reside in a ``test`` directory within each module. Tests are short
methods that call functions in pyxem and compare resulting output values with known
answers. Install necessary dependencies to run the tests::

   pip install --editable .[tests]


To run the tests::

   pytest --cov --pyargs pyxem

The ``--cov`` flag makes `coverage.py <https://coverage.readthedocs.io/en/latest/>`_
print a nice report in the terminal. For an even nicer presentation, you can use
``coverage.py`` directly::

   coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect
the coverage in more detail.

Docstring examples are tested
`with pytest <https://docs.pytest.org/en/stable/doctest.html>`_ as well::

   pytest --doctest-modules --ignore-glob=pyxem/*/tests


Building and writing documentation
==================================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting functionality.
Install necessary dependencies to build the documentation::

   pip install --editable .[doc]

In addition, you will need to download the ``pyxem-demos`` repository and place it in the
doc/tutorial directory. This is necessary to build the documentation.

The easiest way to download the ``pyxem-demos`` repository is to use the ``make demos``
command which will clone the repository into the correct location.::

   cd doc
   make demos

Then build the documentation.

   make html

If the build is successful, the documentation will be available in the
``doc/_build/html`` directory. You can open the created ``index.html`` in the browser
and inspect the documentation in more detail.

.. note::
   If you have already cloned the ``pyxem-demos`` repository, you can use the
   ``make demos`` command again to update the repository to the latest version.

The documentation's HTML pages are built in the ``doc/_build/html`` directory from files
in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ plaintext
markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/pyxem/doc/_build/html/index.html`` in the address bar.

Tips for writing Jupyter notebooks that are meant to be converted to reST text files by
`nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- Use ``_ = ax[0].imshow(...)`` to disable Matplotlib output if a Matplotlib
  command is the last line in a cell.
- Refer to our API reference with this general markdown syntax:
  :code:`:meth:`~.signals.Diffraction2D.get_azimuthal_integral1d`` which will be
  displayed as :meth:`~.signals.Diffraction2D.get_azimuthal_integral1d` or
  :code:`:meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral1d`` to have the full
  path: :meth:`pyxem.signals.Diffraction2D.get_azimuthal_integral1d`
- Reference external APIs via standard markdown like :code:`:class:`hyperspy.api.signals.Signal2D``,
  which will be displayed as :class:`hyperspy.api.signals.Signal2D`.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook
  must be added to the gallery in the README.rst to be included in the
  documentation pages.

Switching between Documentation Versions
----------------------------------------

To make switching between documentation versions easier, we have a version switcher
in the documentation. This switcher is located in the ``doc/_static/switcher.json`` file
or at https://pyxem.readthedocs.io/en/latest/_static/switcher.json.  Because the switcher
points to the latest version of the documentation, any update to the documentation will
be retroactively applied to all previous versions which have the switcher enabled.

To update the switcher, you will need to update the ``doc/_static/switcher.json`` file
with the new version number. This will ensure that the version switcher in the
documentation is up to date.

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/pyxem/actions>`_ to ensure that pyxem
can be installed on Windows, macOS and Linux (Ubuntu). After a successful installation,
the CI server runs the tests. After the tests return no errors, code coverage is
reported to `Coveralls <https://coveralls.io/github/pyxem/pyxem?branch=main>`_.

Making a release
================

We use `GitHub Actions <https://github.com/pyxem/pyxem/actions>`_ to automatically
create a new release. Each time a new tag is pushed to the repository, the CI server
will:

1. Build the documentation.  Each tagged release will be added as a
   `stable <https://docs.readthedocs.io/en/stable/versions.html>`_ build
   to `Read the Docs <https://pyxem.readthedocs.io/en/latest/>`_.
2. Publish a new version of pyxem to `PyPI <https://pypi.org/project/pyxem/>`_.
3. Create a new `GitHub release <https://github.com/pyxem/pyxem/releases>`_.
4. Publish a new version of pyxem to `Zenodo <https://zenodo.org/doi/10.5281/zenodo.2649351>`_.

To make a new release, follow these steps:

1. Update the version number in ``release_info.py``.
2. Update the ``CHANGELOG.rst`` file with the date and new version number and a description of
   the changes.
3. Update the list of contributors in both the ``release_info.py`` and ``.zenodo.json``
   files. __ Make sure that the Zenodo file is valid JSON __.
4. Commit the changes and push them to the repository.
5. Create a new tag with the "v" + version number (e.g. "v0.16.0") and make a new release on GitHub.
6. Wait for the CI server to finish the release process.

Then you can increase the version number in ``release_info.py`` to the next minor version
and add a dev suffix (e.g. "0.17.dev0").

After the new version documentation is public. You should update the doc/_static/switcher.json
file with the new version of the documentation. This will ensure that the version switcher in the
documentation is up to date.

.. note::
   If any of the steps fail, you can restart the CI server by clicking on the "Re-run
   jobs" button on the GitHub Actions page. Sometimes the CI server fails because of
   connection issues to Zenodo or PyPI.