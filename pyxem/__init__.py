# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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


import glob
import logging
import os
import warnings

from hyperspy.io import load
from hyperspy.api import roi

import numpy as np

from natsort import natsorted

from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.generators.library_generator import VectorLibraryGenerator
from diffsims.sims.diffraction_simulation import DiffractionSimulation

from .components.diffraction_component import ElectronDiffractionForwardModel
from .generators.calibration_generator import CalibrationGenerator

from .signals.diffraction1d import Diffraction1D
from .signals.diffraction2d import Diffraction2D
from .signals.diffraction1d import LazyDiffraction1D
from .signals.diffraction2d import LazyDiffraction2D
from .signals.electron_diffraction1d import ElectronDiffraction1D
from .signals.electron_diffraction2d import ElectronDiffraction2D
from .signals.electron_diffraction1d import LazyElectronDiffraction1D
from .signals.electron_diffraction2d import LazyElectronDiffraction2D
from .signals.polar_diffraction2d import PolarDiffraction2D
from .signals.polar_diffraction2d import LazyPolarDiffraction2D

from .signals.crystallographic_map import CrystallographicMap

from .generators.red_intensity_generator1d import ReducedIntensityGenerator1D
from .generators.pdf_generator1d import PDFGenerator1D
from .generators.variance_generator import VarianceGenerator

from .signals.diffraction_variance1d import DiffractionVariance1D
from .signals.diffraction_variance2d import DiffractionVariance2D
from .signals.differential_phase_contrast import DPCBaseSignal, DPCSignal1D, DPCSignal2D
from .signals.diffraction_vectors import DiffractionVectors, DiffractionVectors2D
from .signals.indexation_results import TemplateMatchingResults
from .signals.vdf_image import VDFImage
from .signals.pair_distribution_function1d import PairDistributionFunction1D
from .signals.reduced_intensity1d import ReducedIntensity1D

from pyxem.utils.io_utils import load_mib

from . import release_info

__version__ = release_info.version
__author__ = release_info.author
__copyright__ = release_info.copyright
__credits__ = release_info.credits
__license__ = release_info.license
__maintainer__ = release_info.maintainer
__email__ = release_info.email
__status__ = release_info.status

_logger = logging.getLogger(__name__)
