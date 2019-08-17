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

import numpy as np
from skimage.feature import canny
from skimage.measure import label, regionprops


def apply_transform_to_image(img, transform, center=None):
    """Applies transformation matrix to image and recenters it
    http://docs.sunpy.org/en/stable/_modules/sunpy/image/transform.html
    http://stackoverflow.com/q/20161175
    """

    if center is None:
        center = (np.array(img.shape)[::-1]-1)/2.0
    # shift = (center - center.dot(transform)).dot(np.linalg.inv(transform))
    
    displacement = np.dot(transform, center)
    shift = center - displacement
    
    # order=1; linear interpolation, anything higher may introduce artifacts
    img_tf = interpolation.affine_transform(img, transform, offset=shift, mode="constant", order=1, cval=0.0)
    return img_tf


def affine_transform_ellipse_to_circle(azimuth: float, amplitude: float, inverse=False):
    """Usage: 

    e2c = circle_to_ellipse_affine_transform(azimuth, amplitude):
    np.dot(arr, e2c) # arr.shape == (n, 2)
       or
    apply_transform_to_image(img, e2c)

    http://math.stackexchange.com/q/619037
    """
    sin = np.sin(azimuth)
    cos = np.cos(azimuth)
    sx    = 1 - amplitude
    sy    = 1 + amplitude
    
    # apply in this order
    rot1 = np.array((cos, -sin,  sin, cos)).reshape(2,2)
    scale = np.array((sx, 0, 0, sy)).reshape(2,2)
    rot2 = np.array((cos,  sin, -sin, cos)).reshape(2,2)
    
    composite = rot1.dot(scale).dot(rot2)
    
    if inverse:
        return np.linalg.inv(composite)
    else:
        return composite


def affine_transform_circle_to_ellipse(azimuth: float, amplitude: float):
    """Usage: 

    c2e = circle_to_ellipse_affine_transform(azimuth, amplitude):
    np.dot(arr, c2e) # arr.shape == (n, 2)
       or
    apply_transform_to_image(img, c2e)
    """
    return affine_transform_ellipse_to_circle(azimuth, amplitude, inverse=True)


def apply_stretch_correction(z, center=None, azimuth: float=0, amplitude: float=0):
    """Apply stretch correction to image using calibrated values

    center: list of floats
        pixel coordinates of the center of the direct beam
    azimuth: float
        Direction of the azimuth in degrees with respect to the vertical axis (TODO: check this)
    amplitude: float
        The difference in percent between the long and short axes

    returns:
        (N,N) ndarray
    """
    azimuth_rad = np.radians(azimuth)    # go to radians
    amplitude_pc = amplitude / (2*100)   # as percentage
    tr_mat = affine_transform_ellipse_to_circle(azimuth_rad, amplitude_pc)
    z = apply_transform_to_image(z, tr_mat, center=center)
    return z


def find_stretch_correction(img, sigma):
    """Algorithm to find the stretch correction from powder ring images
    Uses the canny edge detection from scipy

    Based on https://github.com/stefsmeets/instamatic/blob/master/instamatic/processing/stretch_correction.py
    """

    # edge detection
    edges = canny(img, sigma=sigma, low_threshold=None, high_threshold=None)

    # get regionprops
    props = get_ring_props(edges)

    # parse results
    plot_props(edges, props)


def calc_radius_with_distortion(x, y, xc, yc, asym, rot):
    """ calculate the distance of each 2D point from the center (xc, yc) """
    xp = x * np.cos(rot) - y * np.sin(rot)
    yp = x * np.sin(rot) + y * np.cos(rot)
    xcp = xc * np.cos(rot) - yc * np.sin(rot)
    ycp = xc * np.sin(rot) + yc * np.cos(rot)

    return np.sqrt((xp - xcp)**2 + asym * (yp - ycp)**2)


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
    def ring_pattern(pts, scale, amplitude, spread, direct_beam_amplitude,
                     asymmetry, rotation):
        """Calculats a polycrystalline gold diffraction pattern given a set of
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
        ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8 = 0.4247, \
            0.4904, 0.6935, 0.8132, 0.8494, 0.9808, 1.0688, 1.0966
        ring1, ring2, ring3, ring4, ring5, ring6, ring7, ring8 = ring1 * scale, \
            ring2 * scale, ring3 * scale, ring4 * scale, ring5 * scale, \
            ring6 * scale, ring7 * scale, ring8 * scale
        amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8 = 1, 0.44, 0.19, \
            0.16, 0.04, 0.014, 0.038, 0.036

        x = pts[:round(np.size(pts, 0) / 2)]
        y = pts[round(np.size(pts, 0) / 2):]
        Ri = calc_radius_with_distortion(x, y, xcenter, ycenter,
                                         asymmetry, rotation)

        denom = 2 * spread**2
        v0 = direct_beam_amplitude * Ri**-2  # np.exp((-1*(Ri)*(Ri))/d0)
        v1 = amp1 * np.exp((-1 * (Ri - ring1) * (Ri - ring1)) / denom)
        v2 = amp2 * np.exp((-1 * (Ri - ring2) * (Ri - ring2)) / denom)
        v3 = amp3 * np.exp((-1 * (Ri - ring3) * (Ri - ring3)) / denom)
        v4 = amp4 * np.exp((-1 * (Ri - ring4) * (Ri - ring4)) / denom)
        v5 = amp5 * np.exp((-1 * (Ri - ring5) * (Ri - ring5)) / denom)
        v6 = amp6 * np.exp((-1 * (Ri - ring6) * (Ri - ring6)) / denom)
        v7 = amp7 * np.exp((-1 * (Ri - ring7) * (Ri - ring7)) / denom)
        v8 = amp8 * np.exp((-1 * (Ri - ring8) * (Ri - ring8)) / denom)

        return amplitude * (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8).ravel()
    return ring_pattern


def generate_ring_pattern(image_size, mask=False, mask_radius=10, scale=100,
                          amplitude=1000, spread=2, direct_beam_amplitude=500,
                          asymmetry=1, rotation=0):
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
    generated_pattern = ring_pattern(pts, scale, amplitude, spread,
                                     direct_beam_amplitude, asymmetry,
                                     rotation)
    generated_pattern = np.reshape(generated_pattern,
                                   (image_size, image_size))

    if mask == True:
        maskROI = calc_radius_with_distortion(x, y, (image_size - 1) / 2,
                                              (image_size - 1) / 2, 1, 0)
        maskROI[maskROI > mask_radius] = 0
        generated_pattern[maskROI > 0] *= 0

    return generated_pattern
