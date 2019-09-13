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
"""Signal class for Pair Distribution Function (PDF) radial profiles
as a function of distance r.
"""

from hyperspy.signals import Signal1D
import numpy as np


class PDF1D(Signal1D):
    _signal_type = "pdf1d"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

    def normalise_signal(self):
        """
        Normalises the Reduced PDF signal to having a maximum of 1.
        This is applied to each pdf separately in a multidimensional signal.

        """
        size_x = self.data.shape[0]
        size_y = self.data.shape[1]
        size_s = self.data.shape[2]

        shaped_signal = self.data.reshape(size_x * size_y, 1, size_s)

        max_values = np.array(list(map(lambda x: np.max(x), shaped_signal)))
        norm_fac = np.divide(1, max_values)
        norm_fac = norm_fac.reshape(size_x, size_y, 1)

        normalised_data = self.data * norm_fac
        self.data = normalised_data
        return
