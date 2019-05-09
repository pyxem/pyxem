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
from math import sin, cos
from hyperspy.roi import CircleROI, Line2DROI

from pyxem import stack_method
from pyxem.libraries.calibration_library import CalibrationDataLibrary
from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.calibration_utils import call_ring_pattern, \
                                          calc_radius_with_distortion, \
                                          generate_ring_pattern


class CalibrationGenerator():
    """Obtains calibration information from common reference standards.

    Parameters
    ----------
    calibration_data : CalibrationDataLibrary
        The signal of electron diffraction data to be used for calibration.

    """
    def __init__(self, calibration_data):
        # Assign attributes
        self.calibration_data = calibration_data
        self.affine_matrix = None
        self.ring_params = None
        self.diffraction_calibration = None
        self.navigation_calibration = None

    def get_elliptical_distortion(self, mask_radius, scale=100, amplitude=1000,
                                  spread=2, direct_beam_amplitude=500,
                                  asymmetry=1, rotation=0):
        """Determine elliptical distortion of the diffraction pattern.

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
        fit_params : np.array()
            Array of fitting parameters. [scale, amplitude, spread,
                                          direct_beam_amplitude, asymmetry,
                                          rotation].
        affine_matrix : np.array()
            Array defining the affine transformation that corrects for lens
            distortions in the diffraction pattern.

        See Also
        --------
            pyxem.utils.calibration_utils.call_ring_pattern

        """
        # Check that necessary calibration data is provided
        if self.calibration_data.au_x_grating_dp is None:
            raise ValueError("This method requires an Au X-grating diffraction "
                             "pattern to be provided. Please update the "
                             "CalibrationDataLibrary.")
        # Set diffraction pattern variable
        standard_dp = self.calibration_data.au_x_grating_dp
        # Define grid values and center indices for ring pattern evaluation
        image_size = standard_dp.data.shape[0]
        xi = np.linspace(0, image_size - 1, image_size)
        yi = np.linspace(0, image_size - 1, image_size)
        x, y = np.meshgrid(xi, yi)
        xcenter = (image_size - 1) / 2
        ycenter = (image_size - 1) / 2
        # Calculate eliptical parameters
        mask = calc_radius_with_distortion(x, y, (image_size - 1) / 2,
                                           (image_size - 1) / 2, 1, 0)
        # Mask direct beam
        mask[mask > mask_radius] = 0
        standard_dp.data[mask > 0] *= 0
        # Manipulate measured data for fitting
        ref = standard_dp.data[standard_dp.data > 0]
        ref = ref.ravel()
        # Define points for fitting
        pts = np.array([x[standard_dp.data > 0].ravel(),
                        y[standard_dp.data > 0].ravel()]).ravel()
        # Set initial parameters for fitting
        x0 = [scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation]
        # Fit ring pattern to experimental data
        xf, cov = curve_fit(call_ring_pattern(xcenter, ycenter),
                            pts, ref, p0=x0)
        # Set ring fitting parameters to attribute
        self.ring_params = xf
        # Calculate affine transform parameters from fit parameters
        scaling = np.array([[1, 0],
                            [0, xf[4]**-0.5]])

        rotation = np.array([[cos(xf[5]), -sin(xf[5])],
                             [sin(xf[5]),  cos(xf[5])]])

        correction = np.linalg.inv(np.dot(rotation.T,
                                          np.dot(scaling, rotation)))

        affine = np.array([[correction[0,0], correction[0,1], 0.00],
                           [correction[1,0], correction[1,1], 0.00],
                           [0.00, 0.00, 1.00]])
        # Set affine matrix to attribute
        self.affine_matrix = affine

        return affine

    def get_distortion_residuals(self, mask_radius, spread):
        """Obtain residuals for experimental data and distortion corrected data
        with respect to a simulated symmetric ring pattern.

        Parameters
        ----------
        mask_radius : int
            Radius, in pixels, for a mask over the direct beam disc.
        spread : float
            Gaussian spread of each ring in the simulated pattern.

        Returns
        -------
        diff_init : ElectronDiffraction
            Difference between experimental data and simulated symmetric ring
            pattern.
        diff_end : ElectronDiffraction
            Difference between distortion corrected data and simulated symmetric
            ring pattern.
        """
        # Check all required parameters are defined as attributes
        if self.calibration_data.au_x_grating_dp is None:
            raise ValueError("This method requires an Au X-grating diffraction "
                             "pattern to be provided. Please update the "
                             "CalibrationDataLibrary.")
        if self.affine_matrix is None:
            raise ValueError("This method requires a distortion matrix to have "
                             "been determined. Use get_elliptical_distortion "
                             "to determine this matrix.")
        # Set name for experimental data pattern
        dpeg = self.calibration_data.au_x_grating_dp
        ringP = self.ring_params
        size = dpeg.data.shape[0]
        dpref = generate_ring_pattern(image_size=size,
                                      mask=True, mask_radius=mask_radius,
                                      scale=ringP[0],
                                      amplitude=ringP[1],
                                      spread=spread,
                                      direct_beam_amplitude=ringP[3],
                                      asymmetry=1, rotation=ringP[5])
        # Apply distortion corrections to experimental data
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction(dpegs.data.reshape((2,2,size,size)))
        dpegs.apply_affine_transformation(self.affine_matrix,
                                          preserve_range=True,
                                          inplace=True)
        # Calculate residuals to be returned
        diff_init = ElectronDiffraction(dpeg.data - dpref.data)
        diff_end = ElectronDiffraction(dpegs.inav[0,0].data - dpref.data)
        residuals = stack_method([diff_init, diff_end])

        return ElectronDiffraction(residuals)

    def plot_corrected_diffraction_pattern(self, reference_circle=True):
        """Plot the distortion corrected diffraction pattern with an optional
        reference circle.

        Parameters
        ----------
        reference_circle : bool
            If True a CircleROI widget is added to the plot for reference.

        """
        # Check all required parameters are defined as attributes
        if self.calibration_data.au_x_grating_dp is None:
            raise ValueError("This method requires an Au X-grating diffraction "
                             "pattern to be provided. Please update the "
                             "CalibrationDataLibrary.")
        if self.affine_matrix is None:
            raise ValueError("This method requires a distortion matrix to have "
                             "been determined. Use get_elliptical_distortion "
                             "to determine this matrix.")
        # Set name for experimental data pattern
        dpeg = self.calibration_data.au_x_grating_dp
        # Apply distortion corrections to experimental data
        size = dpeg.data.shape[0]
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction(dpegs.data.reshape((2,2,size,size)))
        dpegs.apply_affine_transformation(self.affine_matrix,
                                          preserve_range=True,
                                          inplace=True)
        dpegm = dpegs.mean((0,1))
        # Plot distortion corrected data
        dpegm.plot(cmap='magma', vmax=0.1)
        # add reference circle if specified
        if reference_circle is True:
            circ = CircleROI(cx=128, cy=128, r=53.5, r_inner=0)
            circ.add_widget(dpegm)

    def get_diffraction_calibration(self, mask_length, linewidth):
        """Determine the diffraction pattern pixel size calibration in units of
        reciprocal Angsstroms per pixel.

        Parameters
        ----------
        mask_length : float
            Halfwidth of the region excluded from peak finding around the
            diffraction pattern center.
        linewidth : float
            Width of Line2DROI used to obtain line trace from distortion
            corrected diffraction pattern.

        Returns
        -------
        diff_cal : float
            Diffraction calibration in reciprocal Angstroms per pixel.

        """
        # Check that necessary calibration data is provided
        if self.calibration_data.au_x_grating_dp is None:
            raise ValueError("This method requires an Au X-grating diffraction "
                             "pattern to be provided. Please update the "
                             "CalibrationDataLibrary.")
        if self.affine_matrix is None:
            raise ValueError("This method requires a distortion matrix to have "
                             "been determined. Use get_elliptical_distortion "
                             "to determine this matrix.")
        dpeg = self.calibration_data.au_x_grating_dp
        size = dpeg.data.shape[0]
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction(dpegs.data.reshape((2,2,size,size)))
        dpegs.apply_affine_transformation(self.affine_matrix,
                                          preserve_range=True,
                                          inplace=True)
        dpegm = dpegs.mean((0,1))
        # Define line roi along which to take trace for calibration
        line = Line2DROI(x1=5,y1=5, x2=250,y2=250, linewidth=linewidth)
        # Obtain line trace
        trace = line(dpegm)
        trace = trace.as_signal1D(0)
        # Find peaks in line trace either side of direct beam
        db = (np.sqrt(2)*128) - (5*np.sqrt(2))
        pka = trace.isig[db + mask_length:].find_peaks1D_ohaver()[0]['position']
        pkb = trace.isig[:db - mask_length].find_peaks1D_ohaver()[0]['position']
        # Determine predicted position of 022 peak of Au pattern d022=1.437
        au_pre = db - (self.ring_params[0]/1.437)
        au_post = db + (self.ring_params[0]/1.437)
        # Calculate differences between predicted and measured positions
        prediff = np.abs(pkb - au_pre)
        postdiff = np.abs(pka - au_post)
        # Calculate new calibration value based on most accurate peak positions
        dc = (2/1.437)/(pka[postdiff==min(postdiff)]-pkb[prediff==min(prediff)])
        # Store diffraction calibration value as attribute
        self.diffraction_calibration = dc[0]

        return dc[0]

    def get_navigation_calibration(self, line_roi, x1, x2, n, xspace,
                                   *args, **kwargs):
        """Determine the navigation space pixel size calibration, nm per pixel.

        Parameters
        ----------
        line_roi : Line2DROI
            Line2DROI object along which a profile will be taken to determine
        x1 : float
            Estimate of first X-grating intersection.
        x2 : float
            Estimate of second X-grating intersection.
        n : int
            Number of X-grating squares crossed.
        xspace : float
            Spacing of X-grating in nanometres.

        Returns
        -------
        nav_cal : float
            Navigation calibration in nanometres per pixel.

        """
        # Check that necessary calibration data is provided
        if self.calibration_data.au_x_grating_im is None:
            raise ValueError("This method requires an Au X-grating image to be "
                             "provided. Please update the "
                             "CalibrationDataLibrary.")
        # Obtain line trace
        trace = line_roi(self.calibration_data.au_x_grating_im).as_signal1D(0)
        # Find peaks in line trace
        pk = trace.find_peaks1D_ohaver(*args, **kwargs)[0]['position']
        # Determine peak positions
        dif1 = np.abs(pk - x1)
        dif2 = np.abs(pk - x2)
        # Calculate navigation calibration
        x = (n*xspace)/(pk[dif2==min(dif2)]-pk[dif1==min(dif1)])
        # Store navigation calibration value as attribute
        self.navigation_calibration = x[0]

        return x[0]

    def plot_calibrated_data(self, data_to_plot, *args, **kwargs):
        """ Plot calibrated data for visual inspection.

        Parameters
        ----------
        data_to_plot : string
            Specify the calibrated data to be plotted. Valid options are:
            {'au_x_grating_dp', 'au_x_grating_im', 'moo3_dp', 'moo3_im'}
        """
        # Construct object containing user defined data to plot and set the
        # calibration checking that it is defined.
        if data_to_plot == 'au_x_grating_dp':
            dpeg = self.calibration_data.au_x_grating_dp
            size = dpeg.data.shape[0]
            dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
            dpegs = ElectronDiffraction(dpegs.data.reshape((2,2,size,size)))
            dpegs.apply_affine_transformation(self.affine_matrix,
                                              preserve_range=True,
                                              inplace=True)
            data = dpegs.mean((0,1))
            data.set_diffraction_calibration(self.diffraction_calibration)
        elif data_to_plot == 'au_x_grating_im':
            data = self.calibration_data.au_x_grating_im
        #Plot the data
        data.plot(*args, **kwargs)
