# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from hyperspy.io import load as hyperspyload
from hyperspy.api import roi

import numpy as np

from natsort import natsorted

from .components.diffraction_component import ElectronDiffractionForwardModel

from .generators.diffraction_generator import DiffractionGenerator
from .generators.library_generator import DiffractionLibraryGenerator
from .generators.library_generator import VectorLibraryGenerator
from .generators.calibration_generator import CalibrationGenerator

from .signals.crystallographic_map import CrystallographicMap
from .signals.diffraction_profile import ElectronDiffractionProfile
from .signals.electron_diffraction import ElectronDiffraction
from .signals.diffraction_simulation import DiffractionSimulation
from .signals.diffraction_vectors import DiffractionVectors
from .signals.indexation_results import TemplateMatchingResults
from .signals.vdf_image import VDFImage

from .io_plugins import io_plugins, default_write_ext
from .io_plugins import mib as mib_reader

_logger = logging.getLogger(__name__)


signal_dictionary = {'electron_diffraction':ElectronDiffraction,
                     'template_matching':TemplateMatchingResults,
                     'diffraction_vectors':DiffractionVectors}

def load_mib(filename, scan_size, sum_length=10):
    """Load a medipix hdr/mib file.

    Parameters
    ----------
    filename : string
        File path and name to .hdr file.
    scan_size : int
        Scan size in pixels, allows the function to reshape the array into
        the right shape.
    sum_length : int
        Number of lines to sum over to determine scan fly back location.

    """
    dpt = load_with_reader(filename=filename, reader=mib_reader)
    dpt = ElectronDiffraction(dpt.data.reshape((scan_size, scan_size, 256, 256)))
    trace = dpt.inav[:,0:sum_length].sum((1,2,3))
    edge = np.where(trace==max(trace.data))[0][0]
    if edge==scan_size - 1:
        dp = ElectronDiffraction(dpt.inav[0:edge, 1:])
    else:
        dp = ElectronDiffraction(np.concatenate((dpt.inav[edge + 1:, 1:],
                                                 dpt.inav[0:edge, 1:]), axis=1))

    dp.data = np.flip(dp.data, axis=2)

    return dp

def load(filename,is_ElectronDiffraction=True):
    """
    A wrapper around hyperspy's load function that enables auto-setting signals to ElectronDiffraction
    and correct loading of previously saved ElectronDiffraction, TemplateMatchingResults and DiffractionVectors
    objects

    Parameters
    ----------

    filename : str
        A single filename of a previously saved pyxem object. Other arguments may
        succeed, but will have fallen back on hyperspy load and warn accordingly
    is_ElectronDiffraction : bool
        If the signal is not a pxm saved signal (eg - it's a .blo file), cast to
        an ElectronDiffraction object
    """
    if isinstance(filename,str) == False:
        warnings.warn("filename is not a single string, for clarity consider using hs.load()")
        s = hyperspyload(filename)
        return s

    file_suffix = '.' + filename.split('.')[-1]

    if file_suffix == '.mib':
        raise ValueError('mib files must be loaded directly using pxm.load_mib()')

    if file_suffix in ['.hspy','.blo']: # if True we are loading a signal from a format we know
        s = hyperspyload(filename)
        try:
            s = signal_dictionary[s.metadata.Signal.signal_type](s)
        except KeyError:
            if is_ElectronDiffraction:
                s = ElectronDiffraction(s)
            else:
                warnings.warn("No pyxem functionality used, for clarity consider using hs.load()")
    else:
        warnings.warn("file suffix unknown, for clarity consider using hs.load()")
        s = hyperspyload(filename)

    return s
