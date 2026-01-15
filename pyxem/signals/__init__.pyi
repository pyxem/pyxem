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

from .common_diffraction import CommonDiffraction
from .correlation2d import Correlation2D
from .lazy_correlation2d import LazyCorrelation2D
from .diffraction_variance1d import DiffractionVariance1D
from .diffraction_variance2d import DiffractionVariance2D, ImageVariance
from .diffraction_vectors import DiffractionVectors
from .diffraction_vectors2d import DiffractionVectors2D
from .diffraction_vectors1d import DiffractionVectors1D
from .beam_shift import BeamShift
from .lazy_beam_shift import LazyBeamShift
from .differential_phase_contrast import (
    DPCSignal1D,
    DPCSignal2D,
)
from .lazy_differential_phase_contrast import (
    LazyDPCSignal1D,
    LazyDPCSignal2D,
)
from .diffraction1d import Diffraction1D
from .lazy_diffraction1d import LazyDiffraction1D
from .diffraction2d import Diffraction2D
from .lazy_diffraction2d import LazyDiffraction2D
from .polar_vectors import PolarVectors
from .lazy_polar_vectors import LazyPolarVectors
from .electron_diffraction1d import ElectronDiffraction1D
from .lazy_electron_diffraction1d import LazyElectronDiffraction1D
from .electron_diffraction2d import ElectronDiffraction2D
from .lazy_electron_diffraction2d import LazyElectronDiffraction2D
from .indexation_results import VectorMatchingResults, OrientationMap
from .lazy_indexation_results import LazyOrientationMap
from .pair_distribution_function1d import PairDistributionFunction1D
from .polar_diffraction2d import PolarDiffraction2D
from .lazy_polar_diffraction2d import LazyPolarDiffraction2D
from .power2d import Power2D
from .lazy_power2d import LazyPower2D
from .reduced_intensity1d import ReducedIntensity1D
from .segments import LearningSegment, VDFSegment
from .strain_map import StrainMap
from .correlation1d import Correlation1D
from .lazy_correlation1d import LazyCorrelation1D
from .tensor_field import DisplacementGradientMap
from .lazy_tensor_field import LazyDisplacementGradientMap
from .virtual_dark_field_image import VirtualDarkFieldImage
from .lazy_virtual_dark_field_image import LazyVirtualDarkFieldImage
from .insitu_diffraction2d import InSituDiffraction2D
from .lazy_insitu_diffraction2d import LazyInSituDiffraction2D
from .labeled_diffraction_vectors2d import LabeledDiffractionVectors2D
from .lazy_diffraction_vectors import LazyDiffractionVectors
from .lazy_diffraction_vectors2d import LazyDiffractionVectors2D

__all__ = [
    "CommonDiffraction",
    "Correlation2D",
    "LazyCorrelation2D",
    "BeamShift",
    "LazyBeamShift",
    "DPCBaseSignal",
    "DPCSignal1D",
    "DPCSignal2D",
    "LazyDPCBaseSignal",
    "LazyDPCSignal1D",
    "LazyDPCSignal2D",
    "DiffractionVariance1D",
    "DiffractionVariance2D",
    "ImageVariance",
    "DiffractionVectors",
    "LazyDiffractionVectors",
    "DiffractionVectors1D",
    "DiffractionVectors2D",
    "LazyDiffractionVectors2D",
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
    "LazyDisplacementGradientMap",
    "VirtualDarkFieldImage",
    "LazyVirtualDarkFieldImage",
    "InSituDiffraction2D",
    "LazyInSituDiffraction2D",
    "LabeledDiffractionVectors2D",
    "OrientationMap",
    "LazyOrientationMap",
]
