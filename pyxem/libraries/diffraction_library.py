# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

import pickle
import numpy as np


def load_DiffractionLibrary(filename, safety=False):
    """
    Loads a previously saved diffraction library

    Parameters
    ----------
    filename : str
        The location of the file to be loaded
    safety : bool (defaults to False)
        Unpickling is risky, this variable requires you to acknowledge those risks

    Returns
    -------
    DiffractionLibrary
        Previously saved Library

    See Also
    --------
    DiffractionLibrary.pickle_library()
    """
    if safety:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        raise RuntimeError('Unpickling is risky, turn safety to True if \
        trust the author of this content')


def _get_library_from_angles(library, phase, angle):
    """Finds an element that is orientation within 1e-5 of that specified.

    This is necessary because of floating point round off / hashability. If
    multiple entries satisfy the above criterion a random selection is made.

    Parameters
    ----------
    library : DiffractionLibrary
        The library to be searched
    phase : str
        The phase of interest.
    angle : tuple
        The orientation of interest in the same format as library angle keys.

    Returns
    -------
    library_entries : dict
        Dictionary containing the simulation and associated properties

    """

    for key in library[phase]:
        if np.abs(np.sum(np.subtract(list(key), angle))) < 1e-5:
            return library[phase][key]

    # we haven't found a suitable key
    raise ValueError("It appears that no library entry lies with 1e-5 of the target angle")


class DiffractionLibrary(dict):
    """Maps crystal structure (phase) and orientation to simulated diffraction
    data.

    Attributes
    ----------
    identifiers : list of strings/ints
        A list of phase identifiers referring to different atomic structures.
    structures : list of diffpy.structure.Structure objects.
        A list of diffpy.structure.Structure objects describing the atomic
        structure associated with each phase in the library.
    """

    def __init__(self, *args, **kwargs):
        self.identifiers = None
        self.structures = None

    def get_library_entry(self, phase=None, angle=None):
        """Extracts a single DiffractionLibrary entry.

        Parameters
        ----------
        phase : str
            Key for the phase of interest. If unspecified the choice is random.
        angle : tuple
            Key for the orientation of interest. If unspecified the chois is
            random.

        Returns
        -------
        library_entries : dict
            Dictionary containing the simulation associated with the specified
            phase and orientation with associated properties.
        """

        if phase is not None:
            if angle is not None:
                try:
                    return self[phase][angle]
                except KeyError:
                    return _get_library_from_angles(self, phase, angle)
            else:
                for rotation in self[phase].keys():
                    return self[phase][rotation]
        else:
            if angle is not None:
                raise ValueError("To select a certain angle you must first specify a phase")
            for phase in self.keys():
                for rotation in self[phase].keys():
                    return self[phase][rotation]

    def pickle_library(self, filename):
        """Saves a diffraction library in the pickle format.

        Parameters
        ----------
        filename : str
            The location in which to save the file

        See Also
        --------
        load_DiffractionLibrary()
        """
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
