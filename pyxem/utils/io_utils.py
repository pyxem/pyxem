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

# a lot of stuff depends on this, so we have to create it first

import glob
import logging
import os
import warnings

from hyperspy.io import load as hyperspyload

import numpy as np

from pyxem.signals.diffraction1d import Diffraction1D
from pyxem.signals.diffraction2d import Diffraction2D
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.vdf_image import VDFImage
from pyxem.signals.crystallographic_map import CrystallographicMap
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.indexation_results import TemplateMatchingResults

from pyxem.signals.diffraction1d import LazyDiffraction1D
from pyxem.signals.diffraction2d import LazyDiffraction2D
from pyxem.signals.electron_diffraction1d import LazyElectronDiffraction1D
from pyxem.signals.electron_diffraction2d import LazyElectronDiffraction2D

from pyxem.io_plugins import io_plugins, default_write_ext
from pyxem.io_plugins import mib as mib_reader


def load(filename, lazy=False,
         is_ElectronDiffraction2D=True,
         scan_size=None, sum_length=10):
    """Load data into pyxem objects.

    Wraps hyperspy.load to load various file formats and assigns suitable
    loaded data to pyxem signals.

    Parameters
    ----------

    filename : str
        A single filename of a previously saved pyxem object. Other arguments may
        succeed, but will have fallen back on hyperspy load and warn accordingly
    lazy : bool
        If True the file will be opened lazily, i.e. without actually reading
        the data from the disk until required. Allows datasets much larger than
        available memory to be loaded.
    is_ElectronDiffraction2D : bool
        If the signal is not a pxm saved signal (eg - it's a .blo file), cast to
        an ElectronDiffraction2D object
    scan_size : int
        Scan size in pixels, allows the function to reshape the array into
        the right shape when loading a .mib file.
    sum_length : int
        Number of lines to sum over to determine scan fly back location.

    Returns
    -------
    s : Signal
        A pyxem Signal object containing loaded data.
    """
    if isinstance(filename, str) == True:
        file_suffix = '.' + filename.split('.')[-1]
    else:
        warnings.warn("filename is not a single string, for clarity consider using hs.load()")
        s = hyperspyload(filename, lazy=lazy)
        return s

    if file_suffix == '.mib':
        dpt = load_with_reader(filename=filename, reader=mib_reader)
        dpt = ElectronDiffraction2D(dpt.data.reshape((scan_size, scan_size, 256, 256)))
        trace = dpt.inav[:, 0:sum_length].sum((1, 2, 3))
        edge = np.where(trace == max(trace.data))[0][0]
        if edge == scan_size - 1:
            s = ElectronDiffraction2D(dpt.inav[0:edge, 1:])
        else:
            s = ElectronDiffraction2D(np.concatenate((dpt.inav[edge + 1:, 1:],
                                                      dpt.inav[0:edge, 1:]), axis=1))

            s.data = np.flip(dp.data, axis=2)

    signal_dictionary = {'diffraction1d': Diffraction1D,
                         'diffraction2d': Diffraction2D,
                         'electron_diffraction1d': ElectronDiffraction1D,
                         'electron_diffraction2d': ElectronDiffraction2D,
                         'vdf_image': VDFImage,
                         'template_matching': TemplateMatchingResults,
                         'diffraction_vectors': DiffractionVectors,
                         'crystallographic_map': CrystallographicMap}

    lazy_signal_dictionary = {'diffraction1d': LazyDiffraction1D,
                              'diffraction2d': LazyDiffraction2D,
                              'electron_diffraction1d': LazyElectronDiffraction1D,
                              'electron_diffraction2d': LazyElectronDiffraction2D}

    if file_suffix in ['.hspy', '.blo']:  # if True we are loading a signal from a format we know
        s = hyperspyload(filename, lazy=lazy)
        if lazy:
            try:
                s = lazy_signal_dictionary[s.metadata.Signal.signal_type](s)
            except KeyError:
                if is_ElectronDiffraction2D:
                    s = LazyElectronDiffraction2D(s)
                else:
                    warnings.warn("No pyxem functionality used, for clarity consider using hs.load()")
        else:
            try:
                s = signal_dictionary[s.metadata.Signal.signal_type](s)
            except KeyError:
                if is_ElectronDiffraction2D:
                    s = ElectronDiffraction2D(s)
                else:
                    warnings.warn("No pyxem functionality used, for clarity consider using hs.load()")
    else:
        warnings.warn("file suffix unknown, for clarity consider using hs.load()")
        s = hyperspyload(filename, lazy=lazy)

    return s
