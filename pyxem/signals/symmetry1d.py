# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

"""Signal class for pearson correlation and symmetry coefficient."""

from hyperspy.signals import Signal2D, BaseSignal, Signal1D
import numpy as np


class Symmetry1D(Signal1D):
    _signal_type = "symmetry"

    def __init__(self, *args, **kwargs):
        """Create a Symmetry object from a numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal1D. The first arg should be
            a numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal1D
        """
        super().__init__(*args, **kwargs)

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def get_symmetry_coefficient(self):
        """Return symmetry coefficient from pearson correlation function at all real
        space positions (n from 2 to 10).

        Returns
        -------
        sn: Signal1D
            Symmetry coefficient
        """
        sn = Signal1D(np.zeros(self.axes_manager._navigation_shape_in_array+(11,)))
        sn.isig[2] = self.isig[np.pi]
        sn.isig[3] = (self.isig[2 * np.pi / 3] + self.isig[4 * np.pi / 3]) / 2
        sn.isig[4] = (self.isig[np.pi / 2] + self.isig[3 * np.pi / 2]) / 2
        sn.isig[5] = (self.isig[2 * np.pi / 5] + self.isig[4 * np.pi / 5] + self.isig[6 * np.pi / 5] + self.isig[
            8 * np.pi / 5]) / 4
        sn.isig[6] = (self.isig[np.pi / 3] + self.isig[5 * np.pi / 3]) / 2
        sn.isig[7] = (self.isig[2 * np.pi / 7] + self.isig[4 * np.pi / 7] + self.isig[6 * np.pi / 7] + self.isig[
            8 * np.pi / 7] + self.isig[10 * np.pi / 7] + self.isig[12 * np.pi / 7]) / 6
        sn.isig[8] = (self.isig[np.pi / 4] + self.isig[3 * np.pi / 4] + self.isig[5 * np.pi / 4] + self.isig[
            7 * np.pi / 4]) / 4
        sn.isig[9] = (self.isig[2 * np.pi / 9] + self.isig[4 * np.pi / 9] + self.isig[6 * np.pi / 9] + self.isig[
            8 * np.pi / 9] + self.isig[10 * np.pi / 9] + self.isig[12 * np.pi / 9] + self.isig[14 * np.pi / 9] +
                        self.isig[16 * np.pi / 9]) / 8
        sn.isig[10] = (self.isig[np.pi / 5] + self.isig[3 * np.pi / 5] + self.isig[7 * np.pi / 5] + self.isig[
            9 * np.pi / 5]) / 4
        sn.set_signal_type("symmetry")
        sn.axes_manager.navigation_axes = self.axes_manager.navigation_axes
        sn.axes_manager[-1].name = "Symmetry Order"
        return sn
