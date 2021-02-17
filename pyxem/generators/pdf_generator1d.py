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

"""PDF generator and associated tools."""

import numpy as np

from pyxem.signals import PairDistributionFunction1D
from pyxem.utils.signal import transfer_navigation_axes


class PDFGenerator1D:
    """Generates a PairDistributionFunction1D signal from a specified
    ReducedIntensity1D signal.

    Parameters
    ----------
    signal : ReducedIntensity1D
        A reduced intensity radial profile.
    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal

    def get_pdf(self, s_min, s_max=None, r_min=0, r_max=20, r_increment=0.01):
        """Calculates the pdf from the reduced intensity signal.

        Parameters
        ----------
        s_min : float
            Minimum scattering vector s for the pdf calculation. Note that s is
            defined here as s = 2 sin(theta)/lambda = 1/d.
        s_max : float
            Maximum scattering vector s for the pdf calculation. Note that s is
            defined here as s = 2 sin(theta)/lambda = 1/d.
        r_cutoff : list of float
            A list with the format [<r_min>, <r_max>], which sets the
            limits of the real space axis in the calculated PDF.
        r_increment : float
            Step size in r in the extracted PDF.

        Returns
        -------
        pdf : PDF1D
            A signal of pair distribution functions.
        """
        s_scale = self.signal.axes_manager.signal_axes[0].scale
        if s_max is None:
            s_max = self.signal.axes_manager.signal_axes[0].size * s_scale
            print("s_max set to maximum of signal.")

        r_values = np.arange(r_min, r_max, r_increment)
        r_values = r_values.reshape(1, r_values.size)
        s_limits = [int(s_min / s_scale), int(s_max / s_scale)]

        # check that these aren't out of bounds
        if s_limits[1] > self.signal.axes_manager.signal_axes[0].size:
            raise ValueError(
                "User specified s_max is larger than the maximum "
                "scattering vector magnitude in the data. Please reduce "
                "s_max or use s_max=None to use the full scattering range."
            )
        s_values = np.arange(s_limits[0], s_limits[1], 1) * s_scale
        s_values = s_values.reshape(s_values.size, 1)  # column vector

        limited_red_int = self.signal.isig[s_limits[0] : s_limits[1]].data

        pdf_sine = np.sin(2 * np.pi * np.matmul(s_values, r_values))
        # creates a vector of the pdf
        rpdf = PairDistributionFunction1D(
            8 * np.pi * s_scale * np.matmul(limited_red_int, pdf_sine)
        )

        signal_axis = rpdf.axes_manager.signal_axes[0]
        signal_axis.scale = r_increment
        signal_axis.name = "Radius r"
        signal_axis.units = "$Å$"
        rpdf = transfer_navigation_axes(rpdf, self.signal)

        title = self.signal.metadata.General.title
        rpdf.metadata.General.title = f"Pair distribution function of {title}"

        return rpdf
