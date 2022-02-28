# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

import numpy as np
import pyxem.utils.pixelated_stem_tools as pst
from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.signal1d import Signal1D


class BeamShift(Signal1D):

    _signal_type = "beam_shift"

    def make_linear_plane(self, mask=None):
        """Fit linear planes to the beam shifts, which replaces the original data.

        In many scanning transmission electron microscopes, the center position of the
        diffraction pattern will change as a function of the scan position. This is most
        apparent when scanning over large regions (100+ nanometers). Thus, when working
        with datasets, it is typically necessary to correct for this.
        However, other effects can also affect the apparent center point, like
        diffraction contrast. So while it is possible to correct for the beam shift
        by finding the center position in each diffraction pattern, this can lead to
        features such as interfaces or grain boundaries affecting the centering of the
        diffraction pattern. As the shift caused by the scan system is a slow and
        monotonic, it can often be approximated by fitting linear planes to
        the x- and y- beam shifts.

        In addition, for regions within the scan where the center point of the direct
        beam is hard to ascertain precisely, for example in very thick or heavily
        diffracting regions, a mask can be used to ignore fitting the plane to
        these regions.

        This method does this, and replaces the original beam shift data with these
        fitted planes. The beam shift signal can then be directly used in the
        Diffraction2D.center_direct_beam method.

        Note that for very large regions, this linear plane will probably not
        approximate the beam shift very well. In those cases a higher order plane
        will likely be necessary. Alternatively, a vacuum scan with exactly the same
        scanning parameters should be used.

        Parameters
        ----------
        mask : HyperSpy signal, optional
            Must be the same shape as the navigation dimensions of the beam
            shift signal. The True values will be masked.

        Examples
        --------
        >>> s = pxm.signals.BeamShift(np.random.randint(0, 99, (100, 120, 2)))
        >>> s_mask = hs.signals.Signal2D(np.zeros((100, 120), dtype=bool))
        >>> s_mask.data[20:-20, 20:-20] = True
        >>> s.make_linear_plane(mask=s_mask)

        """
        if self._lazy:
            raise ValueError(
                "make_linear_plane is not implemented for lazy signals, "
                "run compute() first"
            )
        s_shift_x = self.isig[0].T
        s_shift_y = self.isig[1].T
        if mask is not None:
            mask = mask.__array__()
        plane_image_x = pst._get_linear_plane_from_signal2d(s_shift_x, mask=mask)
        plane_image_y = pst._get_linear_plane_from_signal2d(s_shift_y, mask=mask)
        plane_image = np.stack((plane_image_x, plane_image_y), -1)
        self.data = plane_image
        self.events.data_changed.trigger(None)

    def to_dpcsignal(self):
        """Get DPCSignal from the BeamShift signal

        The DPCSignal class is focused on analysing the shifts of the beam, for example
        caused by magnetic or electric fields.

        In practice, the signal and navigation dimensions are switched.

        """
        s_dpc = self.T
        s_dpc.set_signal_type("dpc")
        return s_dpc


class LazyBeamShift(BeamShift, LazySignal):
    _signal_type = "beam_shift"

    pass
