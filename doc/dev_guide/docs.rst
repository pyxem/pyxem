Pyxem's (Complicated) Documentation Build and Preview Workflow
--------------------------------------------------------------

One thing that we've been trying to work on is making the documentation better including worked examples,
tutorials and guides.  The Tutorials in particular are a great way to have full worked examples with real data.
But they are also a lot of work to maintain and keep up to date, additionally in order to ensure that they
don't regress we need to build the documentation and check that everything looks good before merging a PR.

Building documentation can be tedious, however, and it takes time to pull down a PR locally, build the
docs, and then check that everything looks good.  We are now using GitHub Pages to host the documentation
as it gives us more control over the build process that Read the Docs.  However, one thing that
but one thing they lack is a way to preview the documentation before it is merged into the main branch. This is
**not** exactly an easy thing to do so we figured that it would be a good idea to document exactly how we do it
not only for ourselves but also for anyone else who might want to do the same thing.

Building the Documentation
--------------------------

Currently the documentation is built using Sphinx and hosted on GitHub Pages.  We use a shared GitHub
action between multiple hyperspy projects to build the documentation.  This builds the documentation, checks
for broken links and handles caching.  In particular we cache the sphinx-gallery examples, the python environment
and the pooch downloads (data from zenodo used in the examples).

Then we deploy the built documentation.  This goes to one of three places:

1. development deployment: to the ``dev`` folder of the ``gh-pages`` branch of the ``pyxem/pyxem`` repository. This is the most up-to-date version of the
   documentation for anyone who has downloaded and built the package directly from GitHub. It is built every time
   there is a push to the main branch.
2. Release version: to a `vx.x.x` folder in the the `gh-pages` branch of the ``pyxem/pyxem`` repository. This is a versioned copy of the documentation
   for some release. This is built every time there is a new release and a new version tag is created.  It can also
   be built manually if needed using a workflow dispatch event.
3. PR preview: to the ``PR-<pr_number>/`` folder of the ``CSSFRancis/pyxem-docs-staging`` repository. This is a preview of the documentation for a
    pull request. This is built every time there is a new pull request or a push to an existing pull request. When
    the pull request is closed the folder is deleted.  This allows us to preview the documentation before it is
    merged into the main branch.
