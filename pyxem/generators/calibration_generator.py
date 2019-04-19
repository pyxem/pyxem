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

"""Electron diffraction pattern calibration operations.

"""

import numpy as np
from scipy.optimize import curve_fit

from pyxem.utils.calibration_utils import call_ring_pattern, \
                                          calc_radius_with_distortion

class CalibrationGenerator():
    """Obtains calibration information from common reference standards.

    Parameters
    ----------
    signal : ElectronDiffraction
        The signal of electron diffraction data to be used for calibration.
    standard : string
        Identifier for calibration standard used. At present only "au-x-grating"
        is supported.

    """
    def __init__(self,
                 signal,
                 standard='au-x-grating'):
        # Verify calibration standard is recognized and set attribute.
        standard_dict = {
            'au-x-grating': 'au-x-grating',
        }
        if standard in standard_dict:
            self.standard = standard_dict[standard]
        else:
            raise NotImplementedError("The standard `{}` is not recognized. "
                                      "See documentation for available "
                                      "implementations.".format(standard))
        # Check calibration data provided as ElectronDiffraction object.
        if signal is not ElectronDiffraction:
            raise ValueError("Data for calibration must be provided as an "
                             "ElectronDiffraction object.")
        # Set calibration data in attribute after checking standard form.
        if signal.axes_manager.navigation_shape == 1:
            self.signal = signal
        else:
            raise ValueError("Calibration using au-x-grating data requires "
                             "a single diffraction pattern to be provided.")

    def get_diffraction_lens_distortion(self, mask_radius, scale=100,
                                        amplitude=1000, spread=2,
                                        direct_beam_amplitude=500, asymmetry=1,
                                        rotation=0):
        """Determine diffraction pattern calibration and distortions from by
        fitting a polycrystalline gold diffraction pattern to a set of rings.

        Parameters
        ----------
        mask_radius : int
            The radius in pixels for a mask over the direct beam disc
            (the direct beam disc within given radius will be excluded
            from the fit)
        scale : float
            An initial guess for the diffraction calibration
            in 1/Angstrom units
        amplitude : float
            An initial guess for the amplitude of the polycrystalline rings
            in arbitrary units
        spread : float
            An initial guess for the spread within each ring (Gaussian width)
        direct_beam_amplitude : float
            An initial guess for the background intensity from the direct
            beam disc in arbitrary units
        asymmetry : float
            An initial guess for any elliptical asymmetry in the
            pattern (for a perfectly circular pattern asymmetry=1)
        rotation : float
            An initial guess for the rotation of the (elliptical) pattern
            in radians.

        Returns
        -------
        residuals : pxm.ElectronDiffraction()
            Residual between data before and after distortion correct with
            respect to simulated
        fit_params : np.array()
            Array of fitting parameters. [scale, amplitude, spread,
                                          direct_beam_amplitude, asymmetry,
                                          rotation].
        affine_matrix : np.array()


        See Also
        --------
            A utility function generate_ring_pattern is implemented and may be
            used to manually determine appropriate initial parameters.
        """
        standard_dp = self.signal
        image_size = standard_dp.data.shape[0]
        xi = np.linspace(0, image_size - 1, image_size)
        yi = np.linspace(0, image_size - 1, image_size)
        x, y = np.meshgrid(xi, yi)

        mask = calc_radius_with_distortion(x, y, (image_size - 1) / 2,
                                           (image_size - 1) / 2, 1, 0)
        mask[mask > mask_radius] = 0
        standard_dp.data[mask > 0] *= 0

        ref = standard_dp.data[standard_dp.data > 0]
        ref = ref.ravel()

        pts = np.array([x[standard_dp.data > 0].ravel(),
                        y[standard_dp.data > 0].ravel()]).ravel()
        xcenter = (image_size - 1) / 2
        ycenter = (image_size - 1) / 2

        x0 = [scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation]
        xf, cov = curve_fit(call_ring_pattern(xcenter, ycenter), pts, ref, p0=x0)

        return xf

    def get_diffraction_calibration(self):
        """Determine the diffraction pattern pixel size calibration.

        Parameters
        ----------

        """
