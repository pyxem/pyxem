# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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

from hyperspy.api import load as hsload
from hyperspy.api import roi

from pymatgen import Lattice, Structure
import numpy as np

from .diffraction_signal import ElectronDiffraction
from .diffraction_generator import ElectronDiffractionCalculator, DiffractionSimulation
from .library_generator import DiffractionLibraryGenerator
from .diffraction_component import ElectronDiffractionForwardModel
from .scalable_reference_pattern import ScalableReferencePattern
from .utils import mib as mib_reader
from hyperspy.io import load_with_reader

def load(*args, **kwargs):
    """
    Load supported files into a pyxem data structure.
    Supported formats: hspy (HDF5), msa, Gatan dm3, Ripple (rpl+raw), Bruker bcf,
    FEI ser and emi, SEMPER unf, EMD, EDAX spd/spc, tif, Medipix (hdr+mib) and a
    number of image formats.

    Any extra keyword is passed to the corresponding reader. For
    available options see their individual documentation.
    Parameters
    ----------
    filenames :  None, str or list of strings
        The filename to be loaded. If None, a window will open to select
        a file to load. If a valid filename is passed in that single
        file is loaded. If multiple file names are passed in
        a list, a list of objects or a single object containing the data
        of the individual files stacked are returned. This behaviour is
        controlled by the `stack` parameter (see bellow). Multiple
        files can be loaded by using simple shell-style wildcards,
        e.g. 'my_file*.msa' loads all the files that starts
        by 'my_file' and has the '.msa' extension.
    signal_type : {None, "EELS", "EDS_SEM", "EDS_TEM", "", str}
        The acronym that identifies the signal type.
        The value provided may determine the Signal subclass assigned to the
        data.
        If None the value is read/guessed from the file. Any other value
        overrides the value stored in the file if any.
        For electron energy-loss spectroscopy use "EELS".
        For energy dispersive x-rays use "EDS_TEM"
        if acquired from an electron-transparent sample — as it is usually
        the case in a transmission electron  microscope (TEM) —,
        "EDS_SEM" if acquired from a non electron-transparent sample
        — as it is usually the case in a scanning electron  microscope (SEM) —.
        If "" (empty string) the value is not read from the file and is
        considered undefined.
    stack : bool
        If True and multiple filenames are passed in, stacking all
        the data into a single object is attempted. All files must match
        in shape. If each file contains multiple (N) signals, N stacks will be
        created, with the requirement that each file contains the same number
        of signals.
    stack_axis : {None, int, str}
        If None, the signals are stacked over a new axis. The data must
        have the same dimensions. Otherwise the
        signals are stacked over the axis given by its integer index or
        its name. The data must have the same shape, except in the dimension
        corresponding to `axis`.
    new_axis_name : string
        The name of the new axis when `axis` is None.
        If an axis with this name already
        exists it automatically append '-i', where `i` are integers,
        until it finds a name that is not yet in use.
    lazy : {None, bool}
        Open the data lazily - i.e. without actually reading the data from the
        disk until required. Allows opening arbitrary-sized datasets. default
        is `False`.
    print_info: bool
        For SEMPER unf- and EMD (Berkley)-files, if True (default is False)
        additional information read during loading is printed for a quick
        overview.
    downsample : int (1–4095)
        For Bruker bcf files, if set to integer (>=2) (default 1)
        bcf is parsed into down-sampled size array by given integer factor,
        multiple values from original bcf pixels are summed forming downsampled
        pixel. This allows to improve signal and conserve the memory with the
        cost of lower resolution.
    cutoff_at_kV : {None, int, float}
       For Bruker bcf files, if set to numerical (default is None)
       bcf is parsed into array with depth cutoff at coresponding given energy.
       This allows to conserve the memory, with cutting-off unused spectra's
       tail, or force enlargement of the spectra size.
    select_type: {'spectrum', 'image', None}
       For Bruker bcf files, if one of 'spectrum' or 'image' (default is None)
       the loader returns either only hypermap or only SEM/TEM electron images.

    Returns
    -------
    Signal instance or list of signal instances

    Examples
    --------
    Loading a single file providing the signal type:
    >>> d = hs.load('file.dm3', signal_type="EDS_TEM")
    Loading multiple files:
    >>> d = hs.load('file1.dm3','file2.dm3')
    Loading multiple files matching the pattern:
    >>> d = hs.load('file*.dm3')
    Loading (potentially larger than the available memory) files lazily and
    stacking:
    >>> s = hs.load('file*.blo', lazy=True, stack=True)

    See Also
    --------
    io documentation in hyperspy.
    """
    signal = hsload(*args, **kwargs)
    return ElectronDiffraction(**signal._to_dictionary())


def load_mib(filename):
    dpt = load_with_reader(filename=filename, reader=mib_reader)
    dpt = ElectronDiffraction(dpt.data.reshape((256,256,256,256)))
    trace = dpt.inav[:,0:5].sum((1,2,3))
    edge = np.where(trace==max(trace.data))[0][0]
    return ElectronDiffraction(np.concatenate((dpt.inav[edge+1:,1:], dpt.inav[0:edge,1:]), axis=1))
