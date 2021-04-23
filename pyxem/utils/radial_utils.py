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

import math
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from hyperspy.components1d import Polynomial, Gaussian
from hyperspy.signals import Signal2D
from hyperspy.utils.markers import point, line_segment
from hyperspy.misc.utils import isiterable

import pyxem.utils.pixelated_stem_tools as pst


def _centre_comparison(s, steps, step_size, crop_radial_signal=None, angleN=8):
    """
    Compare how the centre position affects the radial average.
    Useful for finding on optimal centre position if one has some
    round feature. Returns a list of image stacks, one image stack
    for each centre position, and the image stack consists of radially
    averaged segments of an image.

    Currently only works with signals with 0 navigation dimensions.

    Parameters
    ----------
    s : HyperSpy 2D signal
    steps : int
    step_size : int
    crop_radial_signal : tuple, optional
    angleN : int, default 8
    """
    if s.axes_manager.navigation_dimension != 0:
        raise ValueError(
            "centre_comparison only works for pixelatedSTEM "
            "signals with 0 navigation dimensions"
        )
    s_list = []
    centre_list = get_centre_position_list(s, steps, step_size)
    for centre_x, centre_y in centre_list:
        s_angle = s.angular_slice_radial_average(
            angleN=angleN,
            centre_x=np.array([centre_x]),
            centre_y=np.array([centre_y]),
            show_progressbar=False,
        )
        if crop_radial_signal is not None:
            s_angle = s_angle.isig[crop_radial_signal[0] : crop_radial_signal[1]]
        s_angle.metadata.add_node("Angle_slice_processing")
        s_angle.metadata.Angle_slice_processing.centre_x = centre_x
        s_angle.metadata.Angle_slice_processing.centre_y = centre_y
        s_list.append(s_angle)
    return s_list


def get_coordinate_of_min(s):
    """Returns the x and y values of the minimum in a signal."""
    z = s.data
    idx = np.argwhere(z == np.min(z))
    x = s.axes_manager[0].index2value(idx[0][1])
    y = s.axes_manager[1].index2value(idx[0][0])
    return (x, y)


def get_centre_position_list(s, steps, step_size):
    """
    Returns a zip of x and y coordinates based on the offset of the center-
    point, and number of steps in each direction and a step-size given in
    pixels. Scale and offset taken from axes_manager.
    """
    scale_x = s.axes_manager.signal_axes[0].scale
    scale_y = s.axes_manager.signal_axes[1].scale
    d1_x = scale_x * steps * step_size
    d2_x = scale_x * (steps + 1) * step_size
    d1_y = scale_y * steps * step_size
    d2_y = scale_y * (steps + 1) * step_size

    x0 = -s.axes_manager[0].offset
    y0 = -s.axes_manager[1].offset
    centre_x_list, centre_y_list = [], []
    range_x = np.arange(x0 - d1_x, x0 + d2_x, step_size * scale_x)
    range_y = np.arange(y0 - d1_y, y0 + d2_y, step_size * scale_y)
    for x in range_x:
        for y in range_y:
            centre_x_list.append(x)
            centre_y_list.append(y)
    centre_list = zip(centre_x_list, centre_y_list)
    return centre_list


def get_optimal_centre_position(
    s, radial_signal_span, steps=3, step_size=1, angleN=8, show_progressbar=True
):
    """
    Find centre position of a ring by using angle sliced radial average.

    Takes signal s, radial span of feature used to determine the centre
    position. Radially averages the feature in angleN segments, for
    each possible centre position. Models this averaged signal
    as a Gaussian and returns array of with the standard deviations
    of these Gaussians.

    Note, the approximate centre position must be set using the offset
    parameter in the signal. The offset parameter is the negative of the
    position shown with s.plot(). So if the centre position in the image is
    x=52 and y=55:
    s.axes_manager.signal_axes[0].offset = -52
    s.axes_manager.signal_axes[1].offset = -55
    This can be checked by replotting the signal, to see if the centre
    position has the values x=0 and y=0.

    Parameters
    ----------
    s : HyperSpy 2D signal
        Only supports signals with no navigation dimensions.
    radial_signal_span : tuple
        Range for finding the circular feature.
    steps : int, default 3
        Number of steps in x/y direction to look for the optimal
        centre position. If the offset is (50, 55), and step_size 1:
        the positions x=(45, 46, 47, 48, ..., 55) and y=(50, 51, 52, ..., 60),
        will be checked.
    step_size : int, default 1
    angleN : int, default 8
        See angular_slice_radial_average for information about angleN.

    Returns
    -------
    Optimal centre position : HyperSpy 2D signal

    Examples
    --------
    >>> s = pxm.dummy_data.get_single_ring_diffraction_signal()
    >>> s.axes_manager.signal_axes[0].offset = -105
    >>> s.axes_manager.signal_axes[1].offset = -67
    >>> import pyxem.utils.radial_utils as ra
    >>> s_ocp = ra.get_optimal_centre_position(s, (35, 45), steps=2)
    >>> centre_pos = ra.get_coordinate_of_min(s_ocp)

    """
    s_list = _centre_comparison(
        s, steps, step_size, crop_radial_signal=radial_signal_span, angleN=angleN
    )

    m_list = []
    for temp_s in tqdm(s_list, disable=(not show_progressbar)):
        temp_s.change_dtype("float64")
        temp_s = temp_s - temp_s.data.min()
        temp_s = temp_s / temp_s.data.max()
        am = temp_s.axes_manager
        g_centre = temp_s.inav[0].data.argmax() * am[-1].scale + am[-1].offset
        g_sigma = 3
        g_A = temp_s.data.max() * 2 * g_sigma

        m = temp_s.create_model()
        g = Gaussian(A=g_A, centre=g_centre, sigma=g_sigma)
        m.append(g)
        m.multifit(show_progressbar=False)
        m_list.append(m)
    s_centre_std_array = _get_offset_image(m_list, s, steps, step_size)
    return s_centre_std_array


def refine_signal_centre_position(
    s, radial_signal_span, refine_step_size=None, **kwargs
):
    """Refine centre position of a diffraction signal by using round feature.

    This function simply calls get_optimal_centre_position with a
    smaller and smaller step_size, giving a more accurate centre position.

    The calculated centre position is stored in the offset value in the
    input signal.

    Parameters
    ----------
    s : HyperSpy 2D signal
        Approximate centre position must be set beforehand, see
        get_optimal_centre_position for more information.
    radial_signal_span : tuple
    refine_step_size : list, default [1, 0.5, 0.25]
    **kwargs
        Passed to get_optimal_centre_position

    See Also
    --------
    get_optimal_centre_position : finds the centre positions for a step_size

    Example
    -------
    >>> import pyxem.utils.radial_utils as ra
    >>> s = pxm.dummy_data.get_single_ring_diffraction_signal()
    >>> s.axes_manager[0].offset, s.axes_manager[1].offset = -103, -69
    >>> ra.refine_signal_centre_position(s, (32., 48.), angleN=4)
    >>> x, y = s.axes_manager[0].offset, s.axes_manager[1].offset

    """
    if refine_step_size is None:
        refine_step_size = [1.0, 0.5, 0.25]
    for step_size in refine_step_size:
        s_centre = get_optimal_centre_position(
            s=s, radial_signal_span=radial_signal_span, step_size=step_size, **kwargs
        )
        x, y = get_coordinate_of_min(s_centre)
        s.axes_manager[0].offset, s.axes_manager[1].offset = -x, -y


def _get_offset_image(model_list, s, steps, step_size):
    """
    Make offset image signal based on difference in Gaussian centre position.

    Creates a Signal2D object of the standard deviation of the
    Gaussians fitted to the radially averaged features, as a
    function of center coordinate.

    Parameters
    ----------
    model_list : list of HyperSpy model objects
    s : HyperSpy 2D signal
    steps : int
    step_size : int

    Returns
    -------
    offset_signal : HyperSpy 2D signal

    """
    s_offset = Signal2D(np.zeros(shape=((steps * 2) + 1, (steps * 2) + 1)))
    am = s.axes_manager.signal_axes

    offset_x0 = -am[0].offset - (steps * step_size)
    offset_y0 = -am[1].offset - (steps * step_size)
    s_offset.axes_manager.signal_axes[0].offset = offset_x0
    s_offset.axes_manager.signal_axes[1].offset = offset_y0
    s_offset.axes_manager.signal_axes[0].scale = step_size
    s_offset.axes_manager.signal_axes[1].scale = step_size
    am_o = s_offset.axes_manager.signal_axes
    for model in model_list:
        x = model.signal.metadata.Angle_slice_processing.centre_x
        y = model.signal.metadata.Angle_slice_processing.centre_y
        std = model.components.Gaussian.centre.as_signal().data.std()
        iX, iY = am_o[0].value2index(x), am_o[1].value2index(y)
        s_offset.data[iY, iX] = std
    return s_offset


def _make_radius_vs_angle_model(
    signal,
    radial_signal_span,
    angleN=15,
    centre_x=None,
    centre_y=None,
    prepeak_range=None,
    show_progressbar=True,
):
    s_ra = signal.angular_slice_radial_average(
        angleN=angleN,
        centre_x=centre_x,
        centre_y=centre_y,
        show_progressbar=show_progressbar,
    )
    s_ra = s_ra.isig[radial_signal_span[0] : radial_signal_span[1]]

    m_ra = s_ra.create_model()

    # Fit 1 order polynomial to edges of data to account for the background
    sa = m_ra.axes_manager.signal_axes[0]
    if prepeak_range is None:
        prepeak_range = (sa.low_value, sa.index2value(4))
    m_ra.set_signal_range(prepeak_range[0], prepeak_range[1])
    m_ra.add_signal_range(sa.index2value(-3), sa.high_value)

    polynomial = Polynomial(1)
    m_ra.append(polynomial)
    m_ra.multifit(show_progressbar=show_progressbar)
    polynomial.set_parameters_not_free()

    m_ra.reset_signal_range()

    # Fit Gaussian to diffraction ring
    centre_initial = (sa.high_value + sa.low_value) * 0.5
    sigma = 3
    A = (s_ra - s_ra.min(axis=1)).data.max() * 2 * sigma

    gaussian = Gaussian(A=A, sigma=sigma, centre=centre_initial)
    gaussian.centre.bmin = sa.low_value
    gaussian.centre.bmax = sa.high_value
    m_ra.append(gaussian)
    return m_ra


def get_radius_vs_angle(
    signal,
    radial_signal_span,
    angleN=15,
    centre_x=None,
    centre_y=None,
    prepeak_range=None,
    show_progressbar=True,
):
    """
    Get radius of a ring as a function of angle.

    This is done by radially averaging angular slices of the image,
    followed by firstly fitting the background around the specified ring,
    then fitting a Gaussian to the remaining intensity in the ring (which
    gets reduced to a peak due to the radial average).

    Useful for finding if a ring is circular or elliptical.
    The centre position has be to set using the offset parameter in
    signal.axes_manager.

    Parameters
    ----------
    signal : HyperSpy 2D signal
        Diffraction image with calibrated centre point.
        This can be set manually, by looking at the diffraction pattern,
        or use a function like radial.get_optimal_centre_position
    radial_signal_span : tuple
    angleN : default 15
    centre_x, centre_y : NumPy arrays
    prepeak_range : tuple, optional
    show_progressbar : default True

    Returns
    -------
    s_centre : HyperSpy 1D signal

    Examples
    --------
    >>> s = pxm.dummy_data.get_single_ring_diffraction_signal()
    >>> s.axes_manager.signal_axes[0].offset = -105
    >>> s.axes_manager.signal_axes[1].offset = -67
    >>> import pyxem.utils.radial_utils as ra
    >>> s_centre = ra.get_radius_vs_angle(s, (35, 45), show_progressbar=False)

    """
    m_ra = _make_radius_vs_angle_model(
        signal=signal,
        radial_signal_span=radial_signal_span,
        angleN=angleN,
        centre_x=centre_x,
        centre_y=centre_y,
        prepeak_range=prepeak_range,
        show_progressbar=show_progressbar,
    )
    m_ra.multifit(fitter="mpfit", bounded=True, show_progressbar=show_progressbar)

    s_centre = m_ra.components.Gaussian.centre.as_signal()
    return s_centre


def get_angle_image_comparison(s0, s1, angleN=12, mask_radius=None):
    """Compare two images by overlaying one on the other in slices.

    This function takes two images, extracts different angular slices
    and combines them into one image.

    Useful for comparing two diffraction images, to see if the rings
    have the same radius.

    Parameters
    ----------
    s0, s1 : HyperSpy 2D Signal
        Both signals need to have the same shape, and no navigation
        dimensions.
    angleN : int, default 12
        Number of angular slices.
    mask_radius : int, optional
        Mask the centre of the image. The default is not to mask anything.
        Useful to mask the most intense parts of the diffraction pattern,
        so the less intense parts can be visualized.

    Returns
    -------
    comparison_signal : HyperSpy 2D

    Examples
    --------
    >>> from pyxem.dummy_data import MakeTestData
    >>> test_data0 = MakeTestData(300, 300)
    >>> test_data0.add_ring(150, 150, 40)
    >>> test_data1 = MakeTestData(300, 300)
    >>> test_data1.add_ring(150, 150, 60)
    >>> s0 = test_data0.signal
    >>> s1 = test_data1.signal
    >>> s0.axes_manager[0].offset, s0.axes_manager[1].offset = -150, -150
    >>> s1.axes_manager[0].offset, s1.axes_manager[1].offset = -150, -150
    >>> import pyxem.utils.radial_utils as ra
    >>> s = ra.get_angle_image_comparison(s0, s1)
    >>> s.plot()

    Mask the inner parts

    >>> s = ra.get_angle_image_comparison(s0, s1, mask_radius=10)

    """
    if s0.axes_manager.shape != s1.axes_manager.shape:
        raise ValueError("s0 and s1 need to have the same shape")
    s = s0.deepcopy()
    angle_array = np.ogrid[0 : 2 * np.pi : (1 + angleN) * 1j]
    for i in range(len(angle_array[:-1])):
        if i % 2:
            angle0, angle1 = angle_array[i : i + 2]
            bool_array = pst._get_angle_sector_mask(s, angle0, angle1)
            s.data[bool_array] = s1.data[bool_array]

    if mask_radius is not None:
        am = s.axes_manager
        mask = pst._make_circular_mask(
            am[0].value2index(0.0),
            am[1].value2index(0.0),
            am[0].size,
            am[1].size,
            mask_radius,
        )
        mask = np.invert(mask)
        s.data *= mask
    return s


def _get_holz_angle(electron_wavelength, lattice_parameter):
    """
    Parameters
    ----------
    electron_wavelength : scalar
        In nanometers
    lattice_parameter : scalar
        In nanometers

    Returns
    -------
    scattering_angle : scalar
        Scattering angle in radians

    Examples
    --------
    >>> import pyxem.utils.radial_utils as ra
    >>> lattice_size = 0.3905 # STO-(001) in nm
    >>> wavelength = 2.51/1000 # Electron wavelength for 200 kV
    >>> angle = ra._get_holz_angle(wavelength, lattice_size)

    """
    k0 = 1.0 / electron_wavelength
    kz = 1.0 / lattice_parameter
    in_root = kz * ((2 * k0) - kz)
    sin_angle = (in_root ** 0.5) / k0
    angle = math.asin(sin_angle)
    return angle


def _scattering_angle_to_lattice_parameter(electron_wavelength, angle):
    """Convert scattering angle data to lattice parameter sizes.

    Parameters
    ----------
    electron_wavelength : float
        Wavelength of the electrons in the electron beam. In nm.
        For 200 kV electrons: 0.00251 (nm)
    angle : NumPy array
        Scattering angle, in radians.

    Returns
    -------
    lattice_parameter : NumPy array
        Lattice parameter, in nanometers

    Examples
    --------
    >>> import pyxem.utils.radial_utils as ra
    >>> angle_list = [0.1, 0.1, 0.1, 0.1] # in radians
    >>> wavelength = 2.51/1000 # Electron wavelength for 200 kV
    >>> lattice_size = ra._scattering_angle_to_lattice_parameter(
    ...     wavelength, angle_list)

    """

    k0 = 1.0 / electron_wavelength
    kz = k0 - (k0 * ((1 - (np.sin(angle) ** 2)) ** 0.5))
    return 1 / kz


def _get_xy_points_from_radius_angle_plot(s_ra):
    x_list = []
    y_list = []
    for angle, radius in zip(s_ra.axes_manager[0].axis, s_ra.data):
        dx = -math.cos(angle) * radius
        dy = -math.sin(angle) * radius
        x_list.append(dx)
        y_list.append(dy)
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    return (x_list, y_list)


def _fit_ellipse_to_xy_points(x, y):
    """Fit an ellipse to a list of x and y points.

    Parameters
    ----------
    x, y : NumPy 2D array

    Returns
    -------
    ellipse_parameters : NumPy array

    """
    xx = x * x
    yy = y * y
    xy = x * y
    ones = np.ones_like(x)
    D = np.vstack((xx, xy, yy, x, y, ones))
    S = np.dot(D, D.T)
    C = np.zeros((6, 6))
    C[0, 2], C[2, 0] = 2, 2
    C[1, 1] = -1
    A, B = la.eig(np.dot(la.inv(S), C))
    i = np.argmax(np.abs(A))
    g = B[:, i]
    return g


def _get_ellipse_parameters(g):
    """
    Parameters
    ----------
    g : NumPy array

    Returns
    -------
    xC, yC : floats
        Centre position
    semi_len0, semi_len1 : floats
        Length of minor and major semi-axis
    rot : float
        Angle between semi_len0 and the positive x-axis. Since semi_len0,
        is not necessarily the longest semi-axis, the rotation will _not_ be
        between the major semi-axis and the positive x-axis.
        In radians, between 0 and pi. The rotation is clockwise, so
        at rot = 0.1 the ellipse will be pointing in the positive x-direction,
        and negative y-direction.
    eccen : float
        Eccentricity of the ellipse

    Note
    ----
    http://mathworld.wolfram.com/Ellipse.html

    """
    a, b, c, d, f, g = g[0], g[1] / 2, g[2], g[3] / 2, g[4] / 2, g[5]
    b2ac = b * b - a * c
    xC = (c * d - b * f) / b2ac
    yC = (a * f - b * d) / b2ac
    frac_top = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    frac_square_bot = (a - c) * (a - c) + 4 * b * b
    semi_len0 = math.sqrt(frac_top / (b2ac * (math.sqrt(frac_square_bot) - (a + c))))
    semi_len1 = math.sqrt(frac_top / (b2ac * (-math.sqrt(frac_square_bot) - (a + c))))
    if b == 0:
        if a < c:
            rot = 0
        else:
            rot = math.pi * 0.5
    else:
        if a < c:
            rot = math.atan((2 * b) / (a - c)) / 2
        else:
            rot = math.pi / 2 + (math.atan((2 * b) / (a - c)) / 2)
    rot = rot % np.pi
    eccen = math.sqrt(1 - ((b * b) / (a * a)))
    return (xC, yC, semi_len0, semi_len1, rot, eccen)


def _get_ellipse_from_parameters(x, y, semi_len0, semi_len1, rot, r_scale=0.05):
    R = np.arange(0, 2 * np.pi, r_scale)
    xx = x + semi_len0 * np.cos(R) * np.cos(rot) - semi_len1 * np.sin(R) * np.sin(rot)
    yy = y + semi_len0 * np.cos(R) * np.sin(rot) + semi_len1 * np.sin(R) * np.cos(rot)
    return (xx, yy)


def _get_marker_list(
    ellipse_parameters, x_list=None, y_list=None, name=None, r_scale=0.05
):
    xC, yC, semi_len0, semi_len1, rot, ecce = _get_ellipse_parameters(
        ellipse_parameters
    )
    xx, yy = _get_ellipse_from_parameters(
        xC, yC, semi_len0, semi_len1, rot, r_scale=r_scale
    )
    marker_list = []
    if x_list is not None:
        for x, y in zip(x_list, y_list):
            point_marker = point(x, y, color="red")
            if name is not None:
                point_marker.name = name + "_" + point_marker.name
            marker_list.append(point_marker)
    for i in range(len(xx)):
        if i == (len(xx) - 1):
            line = line_segment(xx[i], yy[i], xx[0], yy[0], color="green")
        else:
            line = line_segment(xx[i], yy[i], xx[i + 1], yy[i + 1], color="green")
        if name is not None:
            line.name = name + "_" + line.name
        marker_list.append(line)
    return marker_list


def fit_single_ellipse_to_signal(
    s, radial_signal_span, prepeak_range=None, angleN=20, show_progressbar=True
):
    """
    Parameters
    ----------
    s : HyperSpy Signal2D
    radial_signal_span : tuple
    prepeak_range : tuple, optional
    angleN : int, default 20
    show_progressbar : bool, default True

    Returns
    -------
    signal : HyperSpy 2D signal
        Fitted ellipse and fitting points are stored as HyperSpy markers
        in the metadata
    xC, yC : floats
        Centre position
    semi_len0, semi_len1 : floats
        Length of minor and major semi-axis
    rot : float
        Angle between semi_len0 and the positive x-axis. Since semi_len0,
        is not necessarily the longest semi-axis, the rotation will _not_ be
        between the major semi-axis and the positive x-axis.
        In radians, between 0 and pi. The rotation is clockwise, so
        at rot = 0.1 the ellipse will be pointing in the positive x-direction,
        and negative y-direction.
    eccen : float
        Eccentricity of the ellipse

    signal, xC, yC, semi0, semi1, rot, ecc

    Examples
    --------
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> s = pxm.signals.Diffraction2D(np.zeros((200, 220)))
    >>> s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -110
    >>> xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
    >>> ellipse_ring = mdtd._get_elliptical_ring(xx, yy, 0, 0, 50, 70, 0.8)
    >>> s.data += ellipse_ring
    >>> from pyxem.utils.radial_utils import fit_single_ellipse_to_signal
    >>> output = fit_single_ellipse_to_signal(
    ...     s, (40, 80), angleN=30, show_progressbar=False)

    """
    s_ra = get_radius_vs_angle(
        s,
        radial_signal_span,
        angleN=angleN,
        prepeak_range=prepeak_range,
        show_progressbar=show_progressbar,
    )
    x, y = _get_xy_points_from_radius_angle_plot(s_ra)
    ellipse_parameters = _fit_ellipse_to_xy_points(x, y)
    xC, yC, semi0, semi1, rot, ecc = _get_ellipse_parameters(ellipse_parameters)
    marker_list = _get_marker_list(ellipse_parameters, x_list=x, y_list=y)
    s_m = s.deepcopy()
    s_m.add_marker(marker_list, permanent=True, plot_marker=False)
    return s_m, xC, yC, semi0, semi1, rot, ecc


def fit_ellipses_to_signal(
    s, radial_signal_span_list, prepeak_range=None, angleN=20, show_progressbar=True
):
    """
    Parameters
    ----------
    s : HyperSpy Signal2D
    radial_signal_span_list : tuple
    prepeak_range : tuple, optional
    angleN : list or int, default 20
    show_progressbar : bool, default True

    Returns
    -------
    signal : HyperSpy 2D signal
        Fitted ellipse and fitting points are stored as HyperSpy markers
        in the metadata
    xC, yC : floats
        Centre position
    semi_len0, semi_len1 : floats
        Length of the two semi-axes.
    rot : float
        Angle between semi_len0 and the positive x-axis. Since semi_len0,
        is not necessarily the longest semi-axis, the rotation will _not_ be
        between the major semi-axis and the positive x-axis.
        In radians, between 0 and pi. The rotation is clockwise, so
        at rot = 0.1 the ellipse will be pointing in the positive x-direction,
        and negative y-direction.
    eccen : float
        Eccentricity of the ellipse

    Examples
    --------
    >>> from pyxem.dummy_data import make_diffraction_test_data as mdtd
    >>> s = pxm.signals.Diffraction2D(np.zeros((200, 220)))
    >>> s.axes_manager[0].offset, s.axes_manager[1].offset = -100, -110
    >>> xx, yy = np.meshgrid(s.axes_manager[0].axis, s.axes_manager[1].axis)
    >>> ellipse_ring = mdtd._get_elliptical_ring(xx, yy, 0, 0, 50, 70, 0.8)
    >>> s.data += ellipse_ring
    >>> from pyxem.utils.radial_utils import fit_ellipses_to_signal
    >>> output = fit_ellipses_to_signal(
    ...     s, [(40, 80)], angleN=30, show_progressbar=False)

    """
    if not isiterable(angleN):
        angleN = [angleN] * len(radial_signal_span_list)
    else:
        if len(angleN) != len(radial_signal_span_list):
            raise ValueError(
                "angleN and radial_signal_span_list needs to have " "the same length"
            )
    marker_list = []
    ellipse_list = []
    for i, (radial_signal_span, aN) in enumerate(zip(radial_signal_span_list, angleN)):
        s_ra = get_radius_vs_angle(
            s,
            radial_signal_span,
            angleN=aN,
            prepeak_range=prepeak_range,
            show_progressbar=show_progressbar,
        )
        x, y = _get_xy_points_from_radius_angle_plot(s_ra)
        ellipse_parameters = _fit_ellipse_to_xy_points(x, y)
        output = _get_ellipse_parameters(ellipse_parameters)
        ellipse_list.append(output)
        marker_list.extend(
            _get_marker_list(
                ellipse_parameters, x_list=x, y_list=y, name="circle" + str(i)
            )
        )
    s_m = s.deepcopy()
    s_m.add_marker(marker_list, permanent=True, plot_marker=False)
    return s_m, ellipse_list
