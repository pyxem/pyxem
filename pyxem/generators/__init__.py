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

import importlib


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


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "CalibrationGenerator": "calibration_generator",
    "get_DisplacementGradientMap": "displacement_gradient_tensor_generator",
    "get_single_DisplacementGradientTensor": "displacement_gradient_tensor_generator",
    "IndexationGenerator": "indexation_generator",
    "VectorIndexationGenerator": "indexation_generator",
    "TemplateIndexationGenerator": "indexation_generator",
    "ProfileIndexationGenerator": "indexation_generator",
    "AcceleratedIndexationGenerator": "indexation_generator",
    "IntegrationGenerator": "integration_generator",
    "PDFGenerator1D": "pdf_generator1d",
    "ReducedIntensityGenerator1D": "red_intensity_generator1d",
    "SubpixelrefinementGenerator": "subpixelrefinement_generator",
    "VarianceGenerator": "variance_generator",
    "VirtualImageGenerator": "virtual_image_generator",
    "VirtualDarkFieldGenerator": "virtual_image_generator",
}


def __getattr__(name):
    if name in __all__:
        import_path = "pyxem.generators." + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
