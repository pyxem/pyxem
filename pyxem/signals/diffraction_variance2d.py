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

from pyxem.utils.expt_utils import radial_average


class DiffractionVariance2D(Signal2D):
    _signal_type = "diffraction_variance"

    def get_radial_profile(self, inplace=False, **kwargs):
        """Return the radial profile of the diffraction variance signals.

        Returns
        -------
        DiffractionVariance1D
            radial_profile: :obj:`pyxem.signals.DiffractionVariance1D`
            The radial profile of each diffraction variance pattern in the
            DiffractionVariance2D signal.
        **kwargs
            Keyword argument to be passed to the
            py:func:`hyperspy.signal.BaseSignal.map` method.

        See also
        --------
        :func:`pyxem.utils.expt_utils.radial_average`

        Examples
        --------
        .. code-block:: python
            profiles = ed.get_radial_profile()
            profiles.plot()
        """
        radial_profiles = self.map(radial_average, inplace=inplace, **kwargs)
        if inplace:
            # when using inplace, map return None
            radial_profiles = self

        # Assign to the correct class after the signal dimension was reduced
        radial_profiles.set_signal_type(
            radial_profiles.metadata.Signal.signal_type)

        signal_axis = radial_profiles.axes_manager.signal_axes[0]
        signal_axis.offset = 0
        signal_axis.name = 'q'
        signal_axis.units = '$Å^{-1}$'

        if not inplace:
            return radial_profiles


class ImageVariance(Signal2D):
    _signal_type = "image_variance"

    pass
