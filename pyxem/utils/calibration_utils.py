# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import numpy as np


def calc_radius_with_distortion(x, y, xc, yc, asym, rot):
    """ calculate the distance of each 2D point from the center (xc, yc) """
    xp = x * np.cos(rot) - y * np.sin(rot)
    yp = x * np.sin(rot) + y * np.cos(rot)
    xcp = xc * np.cos(rot) - yc * np.sin(rot)
    ycp = xc * np.sin(rot) + yc * np.cos(rot)

    return np.sqrt((xp - xcp) ** 2 + asym * (yp - ycp) ** 2)


def call_ring_pattern(xcenter, ycenter):
    """
    Function to make a call to the function ring_pattern without passing the
    variables directly (necessary for using scipy.optimize.curve_fit).

    Parameters
    ----------
    xcenter : float
        The coordinate (fractional pixel units) of the diffraction
        pattern center in the first dimension
    ycenter : float
        The coordinate (fractional pixel units) of the diffraction
        pattern center in the second dimension

    Returns
    -------
    ring_pattern : function
        A function that calculates a ring pattern given a set of points and
        parameters.

    """

    def ring_pattern(
        pts, scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation
    ):
        """Calculates a polycrystalline gold diffraction pattern given a set of
        pixel coordinates (points). It uses tabulated values of the spacings
        (in reciprocal Angstroms) and relative intensities of rings derived from
        X-ray scattering factors.

        Parameters
        -----------
        pts : 1D array
            One-dimensional array of points (first half as first-dimension
            coordinates, second half as second-dimension coordinates)
        scale : float
            An initial guess for the diffraction calibration
            in 1/Angstrom units
        amplitude : float
            An initial guess for the amplitude of the polycrystalline rings
            in arbitrary units
        spread : float
            An initial guess for the spread within each ring (Gaussian width)
        direct_beam_amplitude : float
            An initial guess for the background intensity from
            the direct beam disc in arbitrary units
        asymmetry : float
            An initial guess for any elliptical asymmetry in the pattern
            (for a perfectly circular pattern asymmetry=1)
        rotation : float
            An initial guess for the rotation of the (elliptical) pattern
            in radians.

        Returns
        -------
        ring_pattern : np.array()
            A one-dimensional array of the intensities of the ring pattern
            at the supplied points.

        """
        rings = [0.4247, 0.4904, 0.6935, 0.8132, 0.8494, 0.9808, 1.0688, 1.0966]
        rings = np.multiply(rings, scale)
        amps = [1, 0.44, 0.19, 0.16, 0.04, 0.014, 0.038, 0.036]

        x = pts[: round(np.size(pts, 0) / 2)]
        y = pts[round(np.size(pts, 0) / 2) :]
        Ri = calc_radius_with_distortion(x, y, xcenter, ycenter, asymmetry, rotation)

        v = []
        denom = 2 * spread ** 2
        v.append(direct_beam_amplitude * Ri ** -2)  # np.exp((-1*(Ri)*(Ri))/d0)
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            v.append(amps[i] * np.exp((-1 * (Ri - rings[i]) * (Ri - rings[i])) / denom))

        return (
            amplitude
            * (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7] + v[8]).ravel()
        )

    return ring_pattern


def generate_ring_pattern(
    image_size,
    mask=False,
    mask_radius=10,
    scale=100,
    amplitude=1000,
    spread=2,
    direct_beam_amplitude=500,
    asymmetry=1,
    rotation=0,
):
    """Calculate a set of rings to model a polycrystalline gold diffraction
    pattern for use in fitting for diffraction pattern calibration.
    It is suggested that the function generate_ring_pattern is used to
    find initial values (initial guess) for the parameters used in
    the function fit_ring_pattern.

    This function is written expecting a single 2D diffraction pattern
    with equal dimensions (e.g. 256x256).

    Parameters
    ----------
    mask : bool
        Choice of whether to use mask or not (mask=True will return a
        specified circular mask setting a region around
        the direct beam to zero)
    mask_radius : int
        The radius in pixels for a mask over the direct beam disc
        (the direct beam disc within given radius will be excluded
        from the fit)
    scale : float
        An initial guess for the diffraction calibration
        in 1/Angstrom units
    image_size : int
        Size of the diffraction pattern to be generated in pixels.
    amplitude : float
        An initial guess for the amplitude of the polycrystalline rings
        in arbitrary units
    spread : float
        An initial guess for the spread within each ring (Gaussian width)
    direct_beam_amplitude : float
        An initial guess for the background intensity from the
        direct beam disc in arbitrary units
    asymmetry : float
        An initial guess for any elliptical asymmetry in the pattern
        (for a perfectly circular pattern asymmetry=1)
    rotation : float
        An initial guess for the rotation of the (elliptical) pattern
        in radians.

    Returns
    -------
    image : np.array()
        Simulated ring pattern with the same dimensions as self.data

    """
    xi = np.linspace(0, image_size - 1, image_size)
    yi = np.linspace(0, image_size - 1, image_size)
    x, y = np.meshgrid(xi, yi)

    pts = np.array([x.ravel(), y.ravel()]).ravel()
    xcenter = (image_size - 1) / 2
    ycenter = (image_size - 1) / 2

    ring_pattern = call_ring_pattern(xcenter, ycenter)
    generated_pattern = ring_pattern(
        pts, scale, amplitude, spread, direct_beam_amplitude, asymmetry, rotation
    )
    generated_pattern = np.reshape(generated_pattern, (image_size, image_size))

    if mask == True:
        maskROI = calc_radius_with_distortion(
            x, y, (image_size - 1) / 2, (image_size - 1) / 2, 1, 0
        )
        maskROI[maskROI > mask_radius] = 0
        generated_pattern[maskROI > 0] *= 0

    return generated_pattern
