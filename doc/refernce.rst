=============
API reference
=============

This reference manual details the public modules, classes, and functions in
pyxem, as generated from their docstrings. Many of the docstrings contain
examples and continued effort to improve these examples is on going.


.. module:: pyxem

The list of top modules (and the load function):

.. autosummary::
    components
    components
    detectors
    generators
    libraries
    signals
    utils

....

Detectors
===============

.. automodule:: pyxem.detectors

.. currentmodule:: pyxem.detectors

.. autosummary::
    GenericFlatDetector
    Medipix256x256Detector
    Medipix515x515Detector

.. autoclass:: GenericFlatDetector
.. autofunction:: Medipix256x256Detector
.. autofunction:: Medipix515x515Detector

....

Generators
===========

.. currentmodule:: pyxem.generators

.. autosummary::
    CalibrationGenerator
    get_DisplacementGradientMap
    get_single_DisplacementGradientTensor
    IndexationGenerator
    VectorIndexationGenerator
    TemplateIndexationGenerator
    ProfileIndexationGenerator
    AcceleratedIndexationGenerator
    IntegrationGenerator
    PDFGenerator1D
    ReducedIntensityGenerator1D
    SubpixelrefinementGenerator
    VarianceGenerator
    VirtualImageGenerator
    VirtualDarkFieldGenerator

.. automodule:: pyxem.generators
    :members:
    :undoc-members:

....

Libraries
=========

.. automodule:: pyxem.libraries

.. autosummary::
    CalibrationDataLibrary


Signals
==========

.. automodule:: pyxem.signals

.. currentmodule:: pyxem.signals

.. autosummary::
    CommonDiffraction
    Correlation2D
    LazyCorrelation2D
    DPCBaseSignal
    DPCSignal1D
    DPCSignal2D
    LazyDPCBaseSignal
    LazyDPCSignal1D
    LazyDPCSignal2D
    DiffractionVariance1D
    DiffractionVariance2D
    ImageVariance
    DiffractionVectors
    DiffractionVectors2D
    Diffraction1D
    LazyDiffraction1D
    Diffraction2D
    LazyDiffraction2D
    ElectronDiffraction1D
    LazyElectronDiffraction1D
    ElectronDiffraction2D
    LazyElectronDiffraction2D
    TemplateMatchingResults
    VectorMatchingResults
    PairDistributionFunction1D
    PolarDiffraction2D
    LazyPolarDiffraction2D
    Power2D
    LazyPower2D
    ReducedIntensity1D
    LearningSegment
    VDFSegment
    StrainMap
    DisplacementGradientMap
    VirtualDarkFieldImage

CommonDiffraction
-------------------

.. currentmodule:: pyxem.signals.CommonDiffraction

.. autoclass:: pyxem.signals.CommonDiffraction
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Correlation2D
------------------

.. currentmodule:: pyxem.signals.Correlation2D

.. autoclass:: pyxem.signals.Correlation2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

BeamShift
------------------

.. currentmodule:: pyxem.signals.BeamShift

.. autoclass:: pyxem.signals.BeamShift
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

DPCSignal1D
------------------

.. currentmodule:: pyxem.signals.DPCSignal1D

.. autoclass:: pyxem.signals.DPCSignal1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

DPCSignal2D
------------------

.. currentmodule:: pyxem.signals.DPCSignal2D

.. autoclass:: pyxem.signals.DPCSignal2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

DiffractionVariance1D
----------------------

.. currentmodule:: pyxem.signals.DiffractionVariance1D

.. autoclass:: pyxem.signals.DiffractionVariance1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

DiffractionVariance2D
----------------------

.. currentmodule:: pyxem.signals.DiffractionVariance2D

.. autoclass:: pyxem.signals.DiffractionVariance2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Diffraction1D
------------------

.. currentmodule:: pyxem.signals.Diffraction1D

.. autoclass:: pyxem.signals.Diffraction1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Diffraction2D
------------------

.. currentmodule:: pyxem.signals.Diffraction2D

.. autoclass:: pyxem.signals.Diffraction2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

ElectronDiffraction1D
----------------------

.. currentmodule:: pyxem.signals.ElectronDiffraction1D

.. autoclass:: pyxem.signals.ElectronDiffraction1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

ElectronDiffraction2D
----------------------

.. currentmodule:: pyxem.signals.ElectronDiffraction2D

.. autoclass:: pyxem.signals.ElectronDiffraction2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

TemplateMatchingResults
------------------------

.. currentmodule:: pyxem.signals.TemplateMatchingResults

.. autoclass:: pyxem.signals.TemplateMatchingResults
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

VectorMatchingResults
----------------------

.. currentmodule:: pyxem.signals.VectorMatchingResults

.. autoclass:: pyxem.signals.VectorMatchingResults
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

PairDistributionFunction1D
---------------------------

.. currentmodule:: pyxem.signals.PairDistributionFunction1D

.. autoclass:: pyxem.signals.PairDistributionFunction1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

PolarDiffraction2D
-------------------

.. currentmodule:: pyxem.signals.PolarDiffraction2D

.. autoclass:: pyxem.signals.PolarDiffraction2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Power2D
------------------

.. currentmodule:: pyxem.signals.Power2D

.. autoclass:: pyxem.signals.Power2D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

ReducedIntensity1D
-------------------

.. currentmodule:: pyxem.signals.ReducedIntensity1D

.. autoclass:: pyxem.signals.ReducedIntensity1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

LearningSegment
------------------

.. currentmodule:: pyxem.signals.LearningSegment

.. autoclass:: pyxem.signals.LearningSegment
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

VDFSegment
------------------

.. currentmodule:: pyxem.signals.VDFSegment

.. autoclass:: pyxem.signals.VDFSegment
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

StrainMap
------------------

.. currentmodule:: pyxem.signals.StrainMap

.. autoclass:: pyxem.signals.StrainMap
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Symmetry1D
------------------

.. currentmodule:: pyxem.signals.symmetry1d.Symmetry1D

.. autoclass:: pyxem.signals.symmetry1d.Symmetry1D
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

DisplacementGradientMap
------------------------

.. currentmodule:: pyxem.signals.DisplacementGradientMap

.. autoclass:: pyxem.signals.DisplacementGradientMap
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

VirtualDarkFieldImage
----------------------

.. currentmodule:: pyxem.signals.VirtualDarkFieldImage

.. autoclass:: pyxem.signals.VirtualDarkFieldImage
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance: