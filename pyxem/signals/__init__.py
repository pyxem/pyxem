# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

from .common_diffraction import CommonDiffraction
from .correlation2d import Correlation2D, LazyCorrelation2D
from .differential_phase_contrast import (
    DPCBaseSignal,
    DPCSignal1D,
    DPCSignal2D,
    LazyDPCBaseSignal,
    LazyDPCSignal1D,
    LazyDPCSignal2D,
)
from .diffraction_variance1d import DiffractionVariance1D
from .diffraction_variance2d import DiffractionVariance2D, ImageVariance
from .diffraction_vectors import DiffractionVectors, DiffractionVectors2D
from .diffraction1d import Diffraction1D, LazyDiffraction1D
from .diffraction2d import Diffraction2D, LazyDiffraction2D
from .electron_diffraction1d import ElectronDiffraction1D, LazyElectronDiffraction1D
from .electron_diffraction2d import ElectronDiffraction2D, LazyElectronDiffraction2D
from .indexation_results import TemplateMatchingResults, VectorMatchingResults
from .pair_distribution_function1d import PairDistributionFunction1D
from .polar_diffraction2d import PolarDiffraction2D, LazyPolarDiffraction2D
from .power2d import Power2D, LazyPower2D
from .reduced_intensity1d import ReducedIntensity1D
from .segments import LearningSegment, VDFSegment
from .strain_map import StrainMap
from .tensor_field import DisplacementGradientMap
from .virtual_dark_field_image import VirtualDarkFieldImage


__all__ = [
    "CommonDiffraction",
    "Correlation2D",
    "LazyCorrelation2D",
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
    "DiffractionVectors2D",
    "Diffraction1D",
    "LazyDiffraction1D",
    "Diffraction2D",
    "LazyDiffraction2D",
    "ElectronDiffraction1D",
    "LazyElectronDiffraction1D",
    "ElectronDiffraction2D",
    "LazyElectronDiffraction2D",
    "TemplateMatchingResults",
    "VectorMatchingResults",
    "PairDistributionFunction1D",
    "PolarDiffraction2D",
    "LazyPolarDiffraction2D",
    "Power2D",
    "LazyPower2D",
    "ReducedIntensity1D",
    "LearningSegment",
    "VDFSegment",
    "StrainMap",
    "DisplacementGradientMap",
    "VirtualDarkFieldImage",
]
