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
"""Signal class for Pair Distribution Function (PDF) radial profiles
as a function of distance r.
"""

from hyperspy.signals import Signal1D

from pyxem.utils.pdf_utils import normalise_pdf_signal_to_max


class PairDistributionFunction1D(Signal1D):
    _signal_type = "pair_distribution_function"

    def normalise_signal(self, s_min=0, inplace=False, *args, **kwargs):
        """
        Normalises the Reduced PDF signal to having a maximum of 1.
        This is applied to each pdf separately in a multidimensional signal.

        Parameters
        ----------
        s_min : float
            The minimum scattering vector s to be considered. Values at lower s
            are ignored. This is to prevent the effect of large oscillations at
            r=0. If not set, the full signal is used.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """
        s_scale = self.axes_manager.signal_axes[0].scale
        index_min = int(s_min / s_scale)

        return self.map(
            normalise_pdf_signal_to_max,
            index_min=index_min,
            inplace=inplace,
            *args,
            **kwargs
        )
