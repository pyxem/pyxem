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

"""Classes to help with generating some more complicated analysis/workflow"""

from ._calibration_generator import CalibrationGenerator
from ._displacement_gradient_tensor_generator import (
    get_DisplacementGradientMap,
    get_single_DisplacementGradientTensor,
)
from ._indexation_generator import (
    IndexationGenerator,
    VectorIndexationGenerator,
    TemplateIndexationGenerator,
    ProfileIndexationGenerator,
    AcceleratedIndexationGenerator,
)
from ._integration_generator import IntegrationGenerator
from ._pdf_generator1d import PDFGenerator1D
from ._red_intensity_generator1d import ReducedIntensityGenerator1D
from ._subpixelrefinement_generator import SubpixelrefinementGenerator
from ._variance_generator import VarianceGenerator
from ._virtual_image_generator import VirtualImageGenerator, VirtualDarkFieldGenerator

__all__ = [
    "CalibrationGenerator",
    "get_DisplacementGradientMap",
    "get_single_DisplacementGradientTensor",
    "IndexationGenerator",
    "VectorIndexationGenerator",
    "TemplateIndexationGenerator",
    "ProfileIndexationGenerator",
    "AcceleratedIndexationGenerator",
    "IntegrationGenerator",
    "PDFGenerator1D",
    "ReducedIntensityGenerator1D",
    "SubpixelrefinementGenerator",
    "VarianceGenerator",
    "VirtualImageGenerator",
    "VirtualDarkFieldGenerator",
]
