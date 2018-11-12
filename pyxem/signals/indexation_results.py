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

import numpy as np
import hyperspy.api as hs
from hyperspy.signal import BaseSignal

from pyxem.utils.sim_utils import peaks_from_best_template, transfer_navigation_axes
from pyxem.utils.indexation_utils import crystal_from_matching_results
from pyxem.utils.indexation_utils import crystal_from_vector_matching
from pyxem.utils.plot import generate_marker_inputs_from_peaks

from pyxem import CrystallographicMap


class TemplateMatchingResults(BaseSignal):
    _signal_type = "matching_results"
    _signal_dimension = 2

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(2)

    def plot_best_matching_results_on_signal(self, signal,
                                             phase, library,
                                             *args, **kwargs):
        """Plot the diffraction vectors on a signal.

        Parameters
        ----------
        signal : ElectronDiffraction
            The ElectronDiffraction signal object on which to plot the peaks.
            This signal must have the same navigation dimensions as the peaks.
        *args :
            Arguments passed to signal.plot()
        **kwargs :
            Keyword arguments passed to signal.plot()
        """
        match_peaks = self.map(peaks_from_best_template,
                               phase=phase, library=library,
                               inplace=False)
        mmx, mmy = generate_marker_inputs_from_peaks(match_peaks)
        signal.plot(*args, **kwargs)
        for mx, my in zip(mmx, mmy):
            m = hs.markers.point(x=mx, y=my, color='red', marker='x')
            signal.add_marker(m, plot_marker=True, permanent=True)

    def get_crystallographic_map(self,
                                 *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching phase and
        orientation at each probe position with corresponding metrics.

        Returns
        -------
        cryst_map : CrystallographicMap
            Crystallographic mapping results containing the best matching phase
            and orientation at each navigation position with associated metrics.

            The Signal at each navigation position is an array of,

                            [phase, np.array((z,x,z)), dict(metrics)]

            which defines the phase, orientation as Euler angles in the zxz
            convention and metrics associated with the matching.

            Metrics for template matching results are
                'correlation'
                'orientation_reliability'
                'phase_reliability'

        """
        # TODO: Add alternative methods beyond highest correlation score.
        crystal_map = self.map(crystal_from_template_matching,
                               inplace=False,
                               *args, **kwargs)

        cryst_map = CrystallographicMap(crystal_map)
        cryst_map = transfer_navigation_axes(cryst_map, self)

        return cryst_map


class VectorMatchingResults(BaseSignal):
    _signal_type = "matching_results"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)

    def get_crystallographic_map(self,
                                 *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching phase and
        orientation at each probe position with corresponding metrics.

        Returns
        -------
        cryst_map : CrystallographicMap
            Crystallographic mapping results containing the best matching phase
            and orientation at each navigation position with associated metrics.

            The Signal at each navigation position is an array of,

                            [phase, np.array((z,x,z)), dict(metrics)]

            which defines the phase, orientation as Euler angles in the zxz
            convention and metrics associated with the matching.

            Metrics for template matching results are
                'match_rate'
                'ehkls'
                'total_error'
                'orientation_reliability'
                'phase_reliability'
        """
        crystal_map = self.map(crystal_from_vector_matching,
                               inplace=False,
                               *args, **kwargs)

        cryst_map = CrystallographicMap(crystal_map)
        cryst_map = transfer_navigation_axes(cryst_map, self)

        return cryst_map
