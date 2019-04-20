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

from hyperspy.signals import BaseSignal

"""
Signal class for diffraction peaks.

These are the 1D equivalent class of a DiffracionVectors class.

They are a list of diffraction peak values in 1D, with dimensions < n | 1 >.
They also contain the attribute "intensity", with the respective peak intensities and dimensions < n | 1 >.
"""


class DiffractionPeaks(BaseSignal):
    """Crystallographic peak finding at each navigation position.

    Attributes
    ----------
    intensity : float
        Intensity associated with each diffraction peak, in the original navigation axes.
    hkls : np.array()
        Array of Miller indices associated with each diffraction vector
        following indexation.
    """
    _signal_type = "diffraction_peaks"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.intensity = None
        self.hkls = None