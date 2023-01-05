# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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
from warnings import warn

from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

from hyperspy.signals import Signal2D

from pyxem.utils.vector_utils import calculate_norms


class DiffractionVectors2D(Signal2D):
    """Crystallographic mapping results containing the best matching crystal
    phase and orientation at each navigation position with associated metrics.

    Attributes
    ----------
    cartesian : np.array()
        Array of 3-vectors describing Cartesian coordinates associated with
        each diffraction vector.
    hkls : np.array()
        Array of Miller indices associated with each diffraction vector
        following indexation.
    """

    _signal_dimension = 2

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)
