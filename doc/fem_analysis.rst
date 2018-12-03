.. _fem_analysis:

========================================
Fluctuation electron microscopy analysis
========================================

:py:meth:`~pixstem.pixelated_stem_class.PixelatedSTEM.fem_analysis`

Fluctuation electron microscopy (FEM) analysis.

.. code-block:: python

    >>> import pixstem.api as ps
    >>> s = ps.dummy_data.get_fem_signal()
    >>> fem_results = s.fem_analysis(centre_x=50, centre_y=50, show_progressbar=False)
    >>> fem_results['V-Omegak'].plot()

.. image:: images/fem_analysis/fem_v_omegak.png
    :scale: 49 %


Plotting function: :py:func:`pixstem.fem_tools.plot_fem`.

.. code-block:: python

    >>> import pixstem.fem_tools at ft
    >>> fig = ft.plot_fem(s, fem_results)

.. image:: images/fem_analysis/fem_full_results.png
    :scale: 49 %


Saving results: :py:func:`pixstem.fem_tools.save_fem`.
This creates a folder with several HyperSpy signals.

.. code-block:: python

    >>> ft.save_fem(fem_results, 'fem_results')


Loading results: :py:func:`pixstem.fem_tools.load_fem`.

.. code-block:: python

    >>> fem_results_loaded = ft.load_fem('fem_results')
