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

from ._common_diffraction import CommonDiffraction
from ._correlation2d import Correlation2D
from ._lazy_correlation2d import LazyCorrelation2D
from ._diffraction_variance1d import DiffractionVariance1D
from ._diffraction_variance2d import DiffractionVariance2D, ImageVariance
from ._diffraction_vectors import DiffractionVectors
from ._diffraction_vectors2d import DiffractionVectors2D
from ._diffraction_vectors1d import DiffractionVectors1D
from ._beam_shift import BeamShift
from ._lazy_beam_shift import LazyBeamShift
from ._differential_phase_contrast import (
    DPCSignal1D,
    DPCSignal2D,
)
from ._lazy_differential_phase_contrast import (
    LazyDPCSignal1D,
    LazyDPCSignal2D,
)
from ._diffraction1d import Diffraction1D
from ._lazy_diffraction1d import LazyDiffraction1D
from ._diffraction2d import Diffraction2D
from ._lazy_diffraction2d import LazyDiffraction2D
from ._polar_vectors import PolarVectors
from ._lazy_polar_vectors import LazyPolarVectors
from ._electron_diffraction1d import ElectronDiffraction1D
from ._lazy_electron_diffraction1d import LazyElectronDiffraction1D
from ._electron_diffraction2d import ElectronDiffraction2D
from ._lazy_electron_diffraction2d import LazyElectronDiffraction2D
from ._indexation_results import VectorMatchingResults, OrientationMap
from ._lazy_indexation_results import LazyOrientationMap
from ._pair_distribution_function1d import PairDistributionFunction1D
from ._polar_diffraction2d import PolarDiffraction2D
from ._lazy_polar_diffraction2d import LazyPolarDiffraction2D
from ._power2d import Power2D
from ._lazy_power2d import LazyPower2D
from ._reduced_intensity1d import ReducedIntensity1D
from ._segments import LearningSegment, VDFSegment
from ._strain_map import StrainMap
from ._correlation1d import Correlation1D
from ._lazy_correlation1d import LazyCorrelation1D
from ._tensor_field import DisplacementGradientMap
from ._lazy_tensor_field import LazyDisplacementGradientMap
from ._virtual_dark_field_image import VirtualDarkFieldImage
from ._lazy_virtual_dark_field_image import LazyVirtualDarkFieldImage
from ._insitu_diffraction2d import InSituDiffraction2D
from ._lazy_insitu_diffraction2d import LazyInSituDiffraction2D
from ._labeled_diffraction_vectors2d import LabeledDiffractionVectors2D
from ._lazy_diffraction_vectors import LazyDiffractionVectors
from ._lazy_diffraction_vectors2d import LazyDiffractionVectors2D

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
