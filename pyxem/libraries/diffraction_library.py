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

import pickle
import numpy as np


def load_DiffractionLibrary(filename, safety=False):
    """Loads a previously saved diffraction library.

    Parameters
    ----------
    filename : str
        The location of the file to be loaded
    safety : bool (defaults to False)
        Unpickling is risky, this variable requires you to acknowledge this.

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


def _get_library_entry_from_angles(library, phase, angles):
    """Finds an element that is orientation within 1e-5 of that specified.

    This is necessary because of floating point round off / hashability. If
    multiple entries satisfy the above criterion a random (the first hit)
    selection is made.

    Parameters
    ----------
    library : DiffractionLibrary
        The library to be searched
    phase : str
        The phase of interest.
    angles : tuple
        The orientation of interest as a tuple of Euler angles following the
        Bunge convention [z, x, z] in degrees.

    Returns
    -------
    orientation_index : int
        Index of the given orientation

    """

    phase_entry = library[phase]
    for orientation_index, orientation in enumerate(phase_entry['orientations']):
        if np.sum(np.abs(np.subtract(orientation, angles))) < 1e-5:
            return orientation_index

    # We haven't found a suitable key
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
    diffraction_generator : DiffractionGenerator
        Diffraction generator used to generate this library.
    reciprocal_radius : float
        Maximum g-vector magnitude for peaks in the library.
    with_direct_beam : bool
        Whether the direct beam included in the library or not.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identifiers = None
        self.structures = None
        self.diffraction_generator = None
        self.reciprocal_radius = 0.0
        self.with_direct_beam = False

    def get_library_entry(self, phase=None, angle=None):
        """Extracts a single DiffractionLibrary entry.

        Parameters
        ----------
        phase : str
            Key for the phase of interest. If unspecified the choice is random.
        angle : tuple
            The orientation of interest as a tuple of Euler angles following the
            Bunge convention [z, x, z] in degrees. If unspecified the choise is
            random (the first hit).

        Returns
        -------
        library_entries : dict
            Dictionary containing the simulation associated with the specified
            phase and orientation with associated properties.

        """

        if phase is not None:
            phase_entry = self[phase]
            if angle is not None:
                orientations = phase_entry['orientations']
                if isinstance(orientations, np.ndarray):
                    orientations = orientations.tolist()
                if angle in orientations:
                    orientation_index = orientations.index(angle)
                else:
                    orientation_index = _get_library_entry_from_angles(self, phase, angle)
            else:
                orientation_index = 0
        else:
            if angle is not None:
                raise ValueError("To select a certain angle you must first specify a phase")
            phase_entry = next(iter(self.values()))
            orientation_index = 0

        return {
            'Sim': phase_entry['simulations'][orientation_index],
            'intensities': phase_entry['intensities'][orientation_index],
            'pixel_coords': phase_entry['pixel_coords'][orientation_index],
            'pattern_norm': np.linalg.norm(phase_entry['intensities'][orientation_index])
        }

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
