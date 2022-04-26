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



from hyperspy.signals import Signal1D
import numpy as np
from fractions import Fraction as frac
from pyxem.utils.correlation_utils import _get_interpolation_matrix, _symmetry_stem


class Correlation1D(Signal1D):
    """Signal class for pearson correlation and symmetry coefficient."""
    
    _signal_type = "correlation"

    def get_symmetry_coefficient(self,
                                 angular_range=0.1,
                                 symmetries=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 method="average",
                                 include_duplicates=False,
                                 normalize=True,
                                 ):
        """Return symmetry coefficient from pearson correlation function at all real
        space positions (n from 2 to 10).

        Parameters
        ----------
        angular_range: float
            The angular range (in rad) to integrate over when calculating the symmetry coefficient.
        symmetries: list
            The list of symmetries to calculate.
        method: "average", "first", "max"
            The method for calculating the Symmetry STEM
        include_duplicates: bool
            If angles which are duplicated should be included.
        normalize: bool
            This normalized by dividing by the number of angles for each symmetry.

        Returns
        -------
        sn: Signal1D
            Symmetry coefficient
        """
        angles = [set(frac(j, i) for j in range(1, i)) for i in symmetries]

        if not include_duplicates:  # remove duplicated symmetries
            already_used = set()
            new_angles = []
            for a in angles:
                new_angles.append(a.difference(already_used))
                already_used = already_used.union(a)
            angles = new_angles
        num_angles = [len(a) for a in angles]

        interp = [_get_interpolation_matrix(a,
                                           angular_range,
                                           num_points=self.axes_manager.signal_axes[0].size,
                                           method=method)
                  for a in angles]
        signals = self.map(_symmetry_stem,
                           interpolation=interp,
                           show_progressbar=True,
                           inplace=False,
                           method=method)
        if method == "max" or method =="first":
            normalize =False
        if normalize:
            signals = np.divide(signals, num_angles)

        signals.axes_manager.signal_axes[-1].name = "Symmetry Order"
        signals.axes_manager.signal_axes[0].scale = 1
        signals.axes_manager.signal_axes[0].name = "Symmetry"
        signals.axes_manager.signal_axes[0].unit = "a.u."
        signals.axes_manager.signal_axes[0].offset = 0
        
        return signals
