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

def load_DiffractionLibrary(filename,safety=False):
    if safety:
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        raise RuntimeError('Unpickling is risky, turn safety to True if \
        trust the author of this content')

def _get_library_from_angles(library,phase,angle):
    """
    This function is designed to find an element that is 'basically' the same
    as the rotation one has asked for. This is needed because
    of floating point round off/hashability.
    """
    residual = 0.1
    for key in library[phase]:
        residual_temp = np.sum(np.subtract(list(key),angle))
        if np.abs(residual_temp) < residual:
            residual = residual_temp
            stored_key = key

    if np.abs(residual) < 1e-5:
        return library[phase][stored_key]
    else:
        raise ValueError("It appears that no library entry lies with 1e-5 of \
        the target angle")


class DiffractionLibrary(dict):
    """Maps crystal structure (phase) and orientation (Euler angles or
    axis-angle pair) to simulated diffraction data.
    """

    def get_library_entry(self,phase=None,angle=None):
        """ Extracts a single library entry for viewing, unspecified layers of
        dict are selected randomly and so this method is not entirely repeatable

        Parameters
        ----------
        Phase : Label for the Phase you are interested in. Randomly chosen
            if False
        Angle : Label for the Angle you are interested in. Randomly chosen
            if False
        """

        if phase is not None:
            if angle is not None:
                try:
                    return self[phase][angle]
                except KeyError:
                    return _get_library_from_angles(self,phase,angle)
            else:
                for rotation in self[phase].keys():
                    return self[phase][rotation]
        else:
            for phase in self.keys():
                for rotation in self[phase].keys():
                    return self[phase][rotation]

    def pickle_library(self,filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
