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

"""Signal class for working with shift of the direct beam."""

from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.signal1d import Signal1D


class BeamShift(Signal1D):

    _signal_type = "beam_shift"

    def make_linear_plane(self, mask=None):
        if self._lazy:
            raise ValueError(
                "make_linear_plane is not implemented for lazy signals, "
                "run compute() first"
            )
        s_shift_x = self.isig[0].T
        s_shift_y = self.isig[1].T
        plane_image_x = _get_linear_plane_from_signal2d(s_shift_x, mask=mask)
        plane_image_y = _get_linear_plane_from_signal2d(s_shift_y, mask=mask)
        plane_image = np.stack((plane_image_x, plane_image_y), -1)
        self.data[:] = plane_image
        self.events.data_changed.trigger(None)

    def to_dpcsignal(self):
        s_dpc = self.T
        s_dpc.set_signal_type("dpc")
        return s_dpc


class LazyBeamShift(BeamShift, LazySignal):
    _signal_type = "beam_shift"

    pass
