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
    """
    Finds an element that is 'basically' the same as the rotation asked for.

    Parameters
    ----------
    library : DiffractionLibrary
        The library to be searched
    phase : str
        The phase to be searched
    angle : tuple
        The target angle, in the same format as the library angle keys

    Returns
    -------
    dict
        Dictionary containing the simulation and associated properties

    Notes
    -----
    This is needed because of floating point round off/hashability.
    'basically' in this context means to within 1e-5
    If more than one angle satisfies this criterion a random solution is returned
    """

    for key in library[phase]:
        if np.abs(np.sum(np.subtract(list(key), angle))) < 1e-5:
            return library[phase][key]

    # we haven't found a suitable key
    raise ValueError("It appears that no library entry lies with 1e-5 of the target angle")


class DiffractionLibrary(dict):
    """
    Maps crystal structure (phase) and orientation to simulated diffraction data.
    """

    def get_library_entry(self, phase=None, angle=None):
        """
        Extracts a single library entry for viewing

        Parameters
        ----------
        Phase : str (default is a random choice)
            label for the phase you are interested in
        Angle : tuple (default is a random choice)
            label for the angle you are interested in.
        Returns
        -------
        dict
            Dictionary containing the simulation and associated properties
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
        """
        Saves a diffraction library in the pickle format

        Parameters
        ----------
        filename : str
            The location in which to save the file

        Returns
        -------
        None

        See Also
        --------
        load_DiffractionLibrary()
        """
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
