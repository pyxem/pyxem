.. _Installation:

Installing pyxem
-----------------

There are a couple of different ways to install pyxem. The easiest way is to use the
hyperspy-bundle. If you don't have much experience with managing python environments, we
highly recommend using the `hyperspy-bundle<https://hyperspy.org/hyperspy-bundle/download.html>`
which is a pre-configured python environment that includes pyxem and all its dependencies.

It includes many other packages that are useful for electron microscopy data analysis, such
as py4dSTEM, AbTEM, and hyperspy. It also includes the Jupyter notebook and Jupyter lab pre-configured
as well as dask-distributed for parallel computing and the ipywidgets for interactive
visualisation.

Installing using Conda/Mamba
----------------------------

If you are more experienced with managing python environments, you can install pyxem
into an existing environment. The easiest way to do this is using conda/mamba.

.. note::
   Installing some of the visualization tools such as ipympl install much better using
   mamba than conda. If you are having trouble installing pyxem, try using mamba instead

```
conda install -c conda-forge pyxem
```

Installing using pip
--------------------

You can also install pyxem using pip if you want. In that case you can install pyxem
using

```
pip install pyxem
```

Installing from source
----------------------

If you want to install the development version of pyxem, you can install it from source.
First clone the repository using

```
git clone https://github.com/pyxem/pyxem.git
```

Then cd into the pyxem directory and install the dependencies using

```
pip install -r requirements.txt
```

Finally, install pyxem using

```
pip install . -e
```
