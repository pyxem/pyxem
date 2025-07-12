# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Signal classes for diffraction data.
"""
import importlib


__all__ = [
    "CommonDiffraction",
    "Correlation2D",
    "LazyCorrelation2D",
    "BeamShift",
    "LazyBeamShift",
    "DPCSignal1D",
    "DPCSignal2D",
    "LazyDPCSignal1D",
    "LazyDPCSignal2D",
    "DiffractionVariance1D",
    "DiffractionVariance2D",
    "ImageVariance",
    "DiffractionVectors",
    "DiffractionVectors1D",
    "DiffractionVectors2D",
    "Diffraction1D",
    "LazyDiffraction1D",
    "LazyDiffraction2D",
    "Diffraction2D",
    "ElectronDiffraction1D",
    "LazyElectronDiffraction1D",
    "LazyElectronDiffraction2D",
    "ElectronDiffraction2D",
    "VectorMatchingResults",
    "PairDistributionFunction1D",
    "PolarDiffraction2D",
    "LazyPolarDiffraction2D",
    "PolarVectors",
    "LazyPolarVectors",
    "Power2D",
    "LazyPower2D",
    "ReducedIntensity1D",
    "LearningSegment",
    "VDFSegment",
    "StrainMap",
    "Correlation1D",
    "LazyCorrelation1D",
    "DisplacementGradientMap",
    "VirtualDarkFieldImage",
    "InSituDiffraction2D",
    "LabeledDiffractionVectors2D",
    "OrientationMap",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "CommonDiffraction": ".common_diffraction",
    "Correlation2D": ".correlation2d",
    "LazyCorrelation2D": ".correlation2d",
    "DiffractionVariance1D": ".diffraction_variance1d",
    "DiffractionVariance2D": ".diffraction_variance2d",
    "ImageVariance": ".diffraction_variance2d",
    "DiffractionVectors": ".diffraction_vectors",
    "DiffractionVectors2D": ".diffraction_vectors2d",
    "DiffractionVectors1D": ".diffraction_vectors1d",
    "BeamShift": ".beam_shift",
    "LazyBeamShift": ".beam_shift",
    "DPCSignal1D": ".differential_phase_contrast",
    "DPCSignal2D": ".differential_phase_contrast",
    "LazyDPCSignal1D": ".differential_phase_contrast",
    "LazyDPCSignal2D": ".differential_phase_contrast",
    "Diffraction1D": ".diffraction1d",
    "LazyDiffraction1D": ".diffraction1d",
    "Diffraction2D": ".diffraction2d",
    "LazyDiffraction2D": ".diffraction2d",
    "PolarVectors": ".polar_vectors",
    "LazyPolarVectors": ".polar_vectors",
    "ElectronDiffraction1D": ".electron_diffraction1d",
    "LazyElectronDiffraction1D": ".electron_diffraction1d",
    "ElectronDiffraction2D": ".electron_diffraction2d",
    "LazyElectronDiffraction2D": ".electron_diffraction2d",
    "VectorMatchingResults": ".indexation_results",
    "OrientationMap": ".indexation_results",
    "PairDistributionFunction1D": ".pair_distribution_function1d",
    "PolarDiffraction2D": ".polar_diffraction2d",
    "LazyPolarDiffraction2D": ".polar_diffraction2d",
    "Power2D": ".power2d",
    "LazyPower2D": ".power2d",
    "ReducedIntensity1D": ".reduced_intensity1d",
    "LearningSegment": ".segments",
    "VDFSegment": ".segments",
    "StrainMap": ".strain_map",
    "Correlation1D": ".correlation1d",
    "LazyCorrelation1D": ".correlation1d",
    "DisplacementGradientMap": ".tensor_field",
    "VirtualDarkFieldImage": ".virtual_dark_field_image",
    "InSituDiffraction2D": ".insitu_diffraction2d",
    "LabeledDiffractionVectors2D": ".labeled_diffraction_vectors2d",
}


def __getattr__(name):
    if name in __all__:
        import_path = "pyxem.signals" + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
