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

"""Indexation generator and associated tools.

"""

import numpy as np
import hyperspy.api as hs

from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.indexation_results import VectorMatchingResults

from pyxem.signals import transfer_navigation_axes

from pyxem.utils.indexation_utils import correlate_library
from pyxem.utils.indexation_utils import index_magnitudes
from pyxem.utils.indexation_utils import match_vectors
from pyxem.utils.indexation_utils import OrientationResult


class PatternMatchGenerator2D():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of measured electron diffraction patterns.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for pattern matching.
    """

    def __init__(self,
                 signal,
                 diffraction_library):
        self.signal = signal
        self.library = diffraction_library

    def correlate(self,
                  n_largest=5,
                  mask=None,
                  *args,
                  **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        mask : Array
            Array with the same size as signal (in navigation) or None
        *args : arguments
            Arguments passed to map().
        **kwargs : arguments
            Keyword arguments passed map().

        Returns
        -------
        matching_results : TemplateMatchingResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern, in the form
            [Library Number , [z, x, z], Correlation Score]

        """
        signal = self.signal
        library = self.library

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # TODO: Add extra methods
        no_extra_methods_yet = True
        if no_extra_methods_yet:
            # adds a normalisation to library
            for phase in library.keys():
                norm_array = np.ones(library[phase]['intensities'].shape[0])  # will store the norms
                for i, intensity_array in enumerate(library[phase]['intensities']):
                    norm_array[i] = np.linalg.norm(intensity_array)
                library[phase]['pattern_norms'] = norm_array  # puts this normalisation into the library

            matches = signal.map(correlate_library,
                                 library=library,
                                 n_largest=n_largest,
                                 mask=mask,
                                 inplace=False,
                                 **kwargs)

        matching_results = TemplateMatchingResults(matches)
        matching_results = transfer_navigation_axes(matching_results, signal)

        return matching_results
