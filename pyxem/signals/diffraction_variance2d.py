# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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
"""Signal class for virtual diffraction contrast images.

"""

from hyperspy.signals import Signal2D

from pyxem.signals.diffraction_variance1d import DiffractionVariance1D
from pyxem.signals import transfer_navigation_axes
from pyxem.utils.expt_utils import radial_average

import numpy as np


class DiffractionVariance2D(Signal2D):
    _signal_type = "diffraction_variance2d"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def get_radial_profile(self, inplace=False, **kwargs):
        """Return the radial profile of the diffraction variance signals.

        Returns
        -------
        DiffractionVariance1D
            radial_profile: :obj:`pyxem.signals.DiffractionVariance1D`
            The radial profile of each diffraction variance pattern in the
            DiffractionVariance2D signal.

        See also
        --------
        :func:`pyxem.utils.expt_utils.radial_average`

        Examples
        --------
        .. code-block:: python
            profiles = ed.get_radial_profile()
            profiles.plot()
        """
        radial_profiles = self.map(radial_average,
                                   inplace=inplace, **kwargs)

        radial_profiles.axes_manager.signal_axes[0].offset = 0
        signal_axis = radial_profiles.axes_manager.signal_axes[0]

        rp = DiffractionVariance1D(radial_profiles.as_signal1D(signal_axis))
        rp = transfer_navigation_axes(rp, self)
        rp_axis = rp.axes_manager.signal_axes[0]
        rp_axis.name = 'q'
        rp_axis.scale = self.axes_manager.signal_axes[0].scale
        rp_axis.units = '$A^{-1}$'

        return rp


class ImageVariance(Signal2D):
    _signal_type = "image_variance"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
