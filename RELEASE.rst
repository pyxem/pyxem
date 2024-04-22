How to make a new release of pyxem
==================================

pyxem versioning tries its best to adhere to `Semantic Versioning
<https://semver.org/spec/v2.0.0.html>`__.
See the `Python Enhancement Proposal (PEP) 440 <https://peps.python.org/pep-0440/>`__
for supported version identifiers.

Preparation
-----------
- Locally, create release branch from the ``main`` branch.

- Run tutorial notebooks and examples in the documentation locally and confirm that they
  produce the expected results.
  From time to time, check the documentation links (``make linkchecker``) and fix any
  broken ones.

- Review the contributor list ``credits`` in ``pyxem/release_info.py`` to ensure all
  contributors are included and sorted correctly.
  Do the same for the Zenodo contributors file ``.zenodo.json``.

- Increment the version number in ``pyxem/release_info.py``.
  Review and clean up ``CHANGELOG.rst`` as per Keep a Changelog.

- Make a PR of the release branch to ``main``.
  Discuss the changelog with others, and make any changes *directly* to the release
  branch.
  Merge the branch into ``main``.

Tag and release
---------------
- Make a tagged release on GitHub.
  The tag target is the ``main`` branch.
  If ``version`` is now "0.16.0", the release name is "pyxem 0.16.0".
  The tag name is "v0.16.0".
  The release body should contain a description of what has changed and a link to the
  changelog.

- Monitor the publish workflow to ensure the release is successfully uploaded to PyPI.
  If the upload fails, the upload workflow can be re-run.

Post-release action
-------------------
- Monitor the `documentation build <https://readthedocs.org/projects/pyxem/builds>`__
  to ensure that the new stable documentation is successfully built from the release.

- Ensure that `Zenodo <https://doi.org/10.5281/zenodo.2649351>`__ displays the new
  release.

- Make a new PR to ``main``, update ``version`` in ``pyxem/release_info.py`` and make
  any updates to this guide if necessary.

- Tidy up GitHub issues and close the corresponding milestone.

- A PR to the conda-forge feedstock will be created by the conda-forge bot.
  Follow the relevant instructions from the conda-forge documentation on updating
  packages, as well as the instructions in the PR.
  Merge after checks pass.
  Monitor the Azure pipeline CI to ensure the release is successfully published to
  conda-forge.

- Update the version switcher in ``doc/_static/switcher.json`` to include the new
  build for the documentation.
