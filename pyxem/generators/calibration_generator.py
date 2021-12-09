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

"""Electron diffraction pattern calibration operations."""

import numpy as np
from scipy.optimize import curve_fit
from math import sin, cos
import matplotlib.pyplot as plt

from hyperspy.roi import CircleROI, Line2DROI
from hyperspy.misc.utils import stack as stack_method
from diffsims.utils.ring_pattern_utils import (
    call_ring_pattern,
    calc_radius_with_distortion,
    generate_ring_pattern,
)

from pyxem.signals import ElectronDiffraction2D
from pyxem.utils.pyfai_utils import get_azimuthal_integrator, _get_setup


class CalibrationGenerator:
    """Obtains calibration information from common reference standards.

    Parameters
    ----------
    calibration_data : CalibrationDataLibrary
        The signal of electron diffraction data to be used for calibration.

    """

    def __init__(
        self, diffraction_pattern=None, grating_image=None, calibration_standard=None
    ):
        """
        Parameters
        ------------
        diffraction_pattern : hyperspy.Signal2D
            Some 2 dimensional signal or numpy array of some
            standard sample used for calibration
        grating_image : array_like
            Some 2 dimensional signal or numpy array of some
            standard sample used for calibration
        calibration_standard : diffpy.structure.Structure
            for calculating the polycrystalline ring spacing
        """
        # Assign attributes
        self.diffraction_pattern = diffraction_pattern
        self.grating_image = grating_image
        self.calibration_standard = calibration_standard
        self.mask = None
        self.ring_params = None
        self.affine_matrix = None
        self.rotation_angle = None
        self.correction_matrix = None
        self.center = None
        self.diffraction_calibration = None
        self.navigation_calibration = None

    def __str__(self):
        information_string = (
            "\n|Calibration Data|\n"
            + "=================\n"
            + "Affine Matrix: "
            + str(self.affine_matrix)
            + "\n"
            + "Rotation Angle:"
            + str(self.rotation_angle)
            + "\n"
            + "Center: "
            + str(self.center)
        )
        return information_string

    def to_ai(self, wavelength, **kwargs):
        sig_shape = np.shape(self.diffraction_pattern)
        unit = "k_A^-1"
        setup = _get_setup(wavelength, unit, self.diffraction_calibration)
        detector, dist, radial_range = setup
        ai = get_azimuthal_integrator(
            detector=detector,
            detector_distance=dist,
            shape=sig_shape,
            center=self.center,
            affine=self.affine_matrix,
            wavelength=wavelength,
            **kwargs,
        )
        return ai

    def get_elliptical_distortion(
        self,
        mask_radius,
        scale=100,
        amplitude=1000,
        spread=2,
        direct_beam_amplitude=500,
        asymmetry=1,
        rotation=0,
        center=None,
    ):
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
        center : None or list
            The center of the diffraction pattern.

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
        if self.diffraction_pattern is None:
            raise ValueError(
                "This method requires a calibration diffraction pattern"
                " to be provided. Please set self.diffraction_pattern equal"
                " to some Signal2D."
            )
        standard_dp = self.diffraction_pattern
        image_size = standard_dp.data.shape[0]
        if center is None:
            center = [(image_size - 1) / 2, (image_size - 1) / 2]
        # Set diffraction pattern variable
        x, y = np.mgrid[0:image_size, 0:image_size]
        radius_map = calc_radius_with_distortion(x, y, center[0], center[1], 1, 0)
        mask = radius_map < mask_radius
        ref = standard_dp.data[~mask]
        fullx, fully = [x, y]
        pts = np.array([fully[~mask], fullx[~mask]]).ravel()
        # Set initial parameters for fitting
        x0 = [scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation]
        # Fit ring pattern to experimental data
        xf, cov = curve_fit(call_ring_pattern(center[0], center[1]), pts, ref, p0=x0)
        # Set ring fitting parameters to attribute
        self.ring_params = xf
        # Calculate affine transform parameters from fit parameters
        scaling = np.array([[1, 0], [0, xf[4] ** -0.5]])
        rotation = np.array([[cos(xf[5]), -sin(xf[5])], [sin(xf[5]), cos(xf[5])]])
        correction = np.linalg.inv(np.dot(rotation.T, np.dot(scaling, rotation)))
        affine = np.array(
            [
                [correction[0, 0], correction[0, 1], 0.00],
                [correction[1, 0], correction[1, 1], 0.00],
                [0.00, 0.00, 1.00],
            ]
        )
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
        diff_init : ElectronDiffraction2D
            Difference between experimental data and simulated symmetric ring
            pattern.
        diff_end : ElectronDiffraction2D
            Difference between distortion corrected data and simulated symmetric
            ring pattern.
        """
        # Check all required parameters are defined as attributes
        if self.diffraction_pattern is None:
            raise ValueError(
                "This method requires an Au X-grating diffraction "
                "pattern to be provided. Please update the "
                "CalibrationDataLibrary."
            )
        if self.affine_matrix is None:
            raise ValueError(
                "This method requires a distortion matrix to have "
                "been determined. Use get_elliptical_distortion "
                "to determine this matrix."
            )
        # Set name for experimental data pattern
        dpeg = self.diffraction_pattern
        ringP = self.ring_params
        size = dpeg.data.shape[0]
        dpref = generate_ring_pattern(
            image_size=size,
            mask=True,
            mask_radius=mask_radius,
            scale=ringP[0],
            amplitude=ringP[1],
            spread=spread,
            direct_beam_amplitude=ringP[3],
            asymmetry=1,
            rotation=ringP[5],
        )
        # Apply distortion corrections to experimental data
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
        dpegs.apply_affine_transformation(
            self.affine_matrix, preserve_range=True, inplace=True
        )
        # Calculate residuals to be returned
        diff_init = ElectronDiffraction2D(dpeg.data - dpref.data)
        diff_end = ElectronDiffraction2D(dpegs.inav[0, 0].data - dpref.data)
        residuals = stack_method([diff_init, diff_end])

        return ElectronDiffraction2D(residuals)

    def plot_corrected_diffraction_pattern(
        self, reference_circle=True, *args, **kwargs
    ):
        """Plot the distortion corrected diffraction pattern with an optional
        reference circle.

        Parameters
        ----------
        reference_circle : bool
            If True a CircleROI widget is added to the plot for reference.
        *args : arguments
            Arguments to be passed to the plot method.
        **kwargs : keyword arguments
            Keyword arguments to be passed to the plot method.

        """
        # Check all required parameters are defined as attributes
        if self.diffraction_pattern is None:
            raise ValueError(
                "This method requires an Au X-grating diffraction "
                "pattern to be provided. Please update the "
                "CalibrationDataLibrary."
            )
        if self.affine_matrix is None:
            raise ValueError(
                "This method requires a distortion matrix to have "
                "been determined. Use get_elliptical_distortion "
                "to determine this matrix."
            )
        # Set name for experimental data pattern
        dpeg = self.diffraction_pattern
        # Apply distortion corrections to experimental data
        size = dpeg.data.shape[0]
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
        dpegs.apply_affine_transformation(
            self.affine_matrix, preserve_range=True, inplace=True
        )
        dpegm = dpegs.mean((0, 1))
        # Plot distortion corrected data
        dpegm.plot(*args, **kwargs)
        # add reference circle if specified
        if reference_circle is True:
            circ = CircleROI(cx=size / 2, cy=size / 2, r=size / 5, r_inner=0)
            circ.add_widget(dpegm)

    def get_diffraction_calibration(self, mask_length, linewidth):
        """Determine the diffraction pattern pixel size calibration in units of
        reciprocal Angstroms per pixel.

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
        if self.diffraction_pattern is None:
            raise ValueError(
                "This method requires an Au X-grating diffraction "
                "pattern to be provided. Please update the "
                "CalibrationDataLibrary."
            )
        if self.affine_matrix is None:
            raise ValueError(
                "This method requires a distortion matrix to have "
                "been determined. Use get_elliptical_distortion "
                "to determine this matrix."
            )
        dpeg = self.diffraction_pattern
        size = dpeg.data.shape[0]
        dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
        dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
        dpegs.apply_affine_transformation(
            self.affine_matrix, preserve_range=True, inplace=True
        )
        dpegm = dpegs.mean((0, 1))
        # Define line roi along which to take trace for calibration
        line = Line2DROI(x1=5, y1=5, x2=size - 6, y2=size - 6, linewidth=linewidth)
        # Obtain line trace
        trace = line(dpegm)
        trace = trace.as_signal1D(0)
        # Find peaks in line trace either side of direct beam
        db = (np.sqrt(2) * (size / 2)) - (5 * np.sqrt(2))
        pka = trace.isig[db + mask_length :].find_peaks1D_ohaver()[0]["position"]
        pkb = trace.isig[: db - mask_length].find_peaks1D_ohaver()[0]["position"]
        # Define Au 220 interplanar spacing (in Angstroms)
        d220 = 1.442
        # Determine predicted position of Au 220 peak
        au_pre = db - (self.ring_params[0] / d220)
        au_post = db + (self.ring_params[0] / d220)
        # Calculate differences between predicted and measured positions
        prediff = np.abs(pkb - au_pre)
        postdiff = np.abs(pka - au_post)
        # Calculate new calibration value based on most accurate peak positions
        dc = (2 / d220) / (
            pka[postdiff == min(postdiff)] - pkb[prediff == min(prediff)]
        )
        # Store diffraction calibration value as attribute
        self.diffraction_calibration = dc[0]

        return dc[0]

    def get_navigation_calibration(self, line_roi, x1, x2, n, xspace, *args, **kwargs):
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
        *args : arguments
            Arguments to be passed to the find_peaks1D method.
        **kwargs : keyword arguments
            Keyword arguments to be passed to the find_peaks1D method.

        Returns
        -------
        nav_cal : float
            Navigation calibration in nanometres per pixel.

        """
        # Check that necessary calibration data is provided
        if self.grating_image is None:
            raise ValueError(
                "This method requires an Au X-grating image to be "
                "provided. Please update the "
                "CalibrationDataLibrary."
            )
        # Obtain line trace
        trace = line_roi(self.grating_image).as_signal1D(0)
        # Find peaks in line trace
        pk = trace.find_peaks1D_ohaver(*args, **kwargs)[0]["position"]
        # Determine peak positions
        dif1 = np.abs(pk - x1)
        dif2 = np.abs(pk - x2)
        # Calculate navigation calibration
        x = (n * xspace) / (pk[dif2 == min(dif2)] - pk[dif1 == min(dif1)])
        # Store navigation calibration value as attribute
        self.navigation_calibration = x[0]
        return x[0]

    def get_rotation_calibration(self, real_line, reciprocal_line):
        """Determine the rotation between real and reciprocal space coordinates.

        Parameters
        ----------
        real_line : Line2DROI
            Line2DROI object drawn along known direction in real space.
        reciprocal_line : Line2DROI
            Line2DROI object drawn along known direction in reciprocal space.

        Returns
        -------
        rotation_angle : float
            Rotation angle in degrees.
        """
        # Calculate rotation angle and store as attribute
        self.rotation_angle = real_line.angle() - reciprocal_line.angle()
        # Return rotation angle calibration
        return self.rotation_angle

    def get_correction_matrix(self):
        """Determine the transformation matrix required to correct for
        diffraction pattern distortions and/or rotation between real and
        reciprocal space coordinates.

        Returns
        -------
        correction_matrix : np.array()
            Array defining the affine transformation that corrects for lens
            distortions in the diffraction pattern.

        """
        if self.affine_matrix is None and self.rotation_angle is None:
            raise ValueError(
                "This method requires either an affine matrix to "
                "correct distortion or a rotation angle between "
                "real and reciprocal space to have been "
                "determined. Please determine these parameters."
            )
        # Only distortion correction case
        elif self.rotation_angle is None:
            correction_matrix = self.affine_matrix
        # Only rotation correction case
        elif self.affine_matrix is None:
            theta = self.rotation_angle / 180 * np.pi
            correction_matrix = np.array(
                [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
            )
        # Otherwise both corrections required, distortion applied first
        else:
            theta = self.rotation_angle / 180 * np.pi
            rotation_matrix = np.array(
                [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
            )
            correction_matrix = np.matmul(rotation_matrix, self.affine_matrix)
        # Set the correction matrix attribute
        self.correction_matrix = correction_matrix
        # Return the correction matrix
        return correction_matrix

    def plot_calibrated_data(
        self, data_to_plot, line=None, unwrap=False, *args, **kwargs
    ):  # pragma: no cover
        """Plot calibrated data for visual inspection.

        Parameters
        ----------
        data_to_plot : str
            Specify the calibrated data to be plotted. Valid options are:
            {'au_x_grating_dp', 'au_x_grating_im', 'moo3_dp', 'moo3_im',
            'rotation_overlay'}
        line : :obj:`hyperspy.roi.Line2DROI`
            An optional Line2DROI object, as detailed in HyperSpy, to be added
            as a widget to the calibration data plot and the trace plotted
            interactively.
        *args : arguments
            Arguments to be passed to the plot method.
        **kwargs : keyword arguments
            Keyword arguments to be passed to the plot method.
        """
        # Construct object containing user defined data to plot and set the
        # calibration checking that it is defined.
        if data_to_plot == "au_x_grating_dp":
            dpeg = self.diffraction_pattern
            size = dpeg.data.shape[0]
            if self.correction_matrix is None:
                self.get_correction_matrix()
            dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
            dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
            dpegs.apply_affine_transformation(
                self.correction_matrix, preserve_range=True, inplace=True
            )
            data = dpegs.mean((0, 1))
            data.set_diffraction_calibration(self.diffraction_calibration)
            # Plot the calibrated diffraction data
            data.plot(*args, **kwargs)
        elif data_to_plot == "au_x_grating_im":
            data = self.diffraction_pattern
            # Plot the calibrated image data
            data.plot(*args, **kwargs)
        elif data_to_plot == "moo3_dp":
            dpeg = self.calibration_data.moo3_dp
            size = dpeg.data.shape[0]
            if self.correction_matrix is None:
                self.get_correction_matrix()
            dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
            dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
            dpegs.apply_affine_transformation(
                self.correction_matrix, preserve_range=True, inplace=True
            )
            data = dpegs.mean((0, 1))
            data.set_diffraction_calibration(self.diffraction_calibration)
            # Plot the calibrated diffraction data
            data.plot(*args, **kwargs)
        elif data_to_plot == "moo3_im":
            data = self.calibration_data.moo3_im
            # Plot the calibrated image data
            data.plot(*args, **kwargs)
        elif data_to_plot == "rotation_overlay":
            dpeg = self.calibration_data.moo3_dp
            size = dpeg.data.shape[0]
            if self.correction_matrix is None:
                self.get_correction_matrix()
            dpegs = stack_method([dpeg, dpeg, dpeg, dpeg])
            dpegs = ElectronDiffraction2D(dpegs.data.reshape((2, 2, size, size)))
            dpegs.apply_affine_transformation(
                self.correction_matrix, preserve_range=True, inplace=True
            )
            dp = dpegs.mean((0, 1))
            im = self.calibration_data.moo3_im.rebin(dp.data.shape)
            stack1 = np.zeros((dp.data.shape[0], dp.data.shape[1], 3))
            stack1[:, :, 0] = dp.data / (0.05 * dp.data.max())
            stack1[:, :, 2] = im.data / im.data.max()
            plt.figure(1)
            plt.imshow(stack1)
        if line:
            line.add_widget(data, axes=data.axes_manager.signal_axes)
            trace = line.interactive(data, navigation_signal="same")
            trace.plot()
            return trace
