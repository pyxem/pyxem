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
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv


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


def solve_ellipse(img, mask=None, interactive=False, num_points=500, suspected_radius=None):
    """Takes a 2-d array and allows you to solve for the equivalent ellipse. Everything is done in array coord.

    Fitzgibbon, A. W., Fisher, R. B., Hill, F., & Eh, E. (1999). Direct Least Squres Fitting of Ellipses.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5), 476–480.
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    Parameters
    ----------
    img : array-like
        Image with an intense ring such as weith an amorpohus materials or poly crystalline one.
    interactive: bool
        Allows you to pick points for the ellipse instead of taking the top 2000 points
    Returns
    ----------
    center: array-like
        In cartesian coordinates or (x,y)!!!!!! arrays are in y,x
    affine: 3x3 matrix
        The affine matrix defining the elliptical distortion
    """
    print(img)
    def fit_ellipse(x, y):
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))   # Design matrix [x^2, xy, y^2, x,y,1]
        S = np.dot(D.T, D)  # Scatter Matrix
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))  # eigen decomposition to solve constrained minimization problem
        n = np.argmax(np.abs(E))   # maximum eigenvalue solution
        a = V[:, n]
        print("a is:", a)
        return a

    def ellipse_center(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        denom = b**2 - (4 * a * c)
        x0 = (2 * c * d - b * e) / denom
        y0 = (2 * a * e - b * d) / denom
        return np.array([x0, y0])

    def ellipse_axis_length(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        denom = b**2 - (4 * a * c)
        num1 = np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) *
                       ((a+c) - np.sqrt((a - c) ** 2 + b ** 2)))
        num2 = np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c) * f) *
                       ((a+c) + np.sqrt((a - c) ** 2 + b ** 2)))
        axis1 = abs(num1/denom)
        axis2 = abs(num2/denom)

        return np.sort([axis1, axis2])[::-1]

    def ellipse_angle_of_rotation(ellipse_parameters):
        a, b, c, d, e, f = ellipse_parameters
        b, d, e = b/2, d/2, e/3
        if b == 0:
            if a > c:
                return 0
            else:
                return np.pi / 2
        else:
            if a < c:
                ang = .5 * invcot((a-c)/(2*b))
                if (a < 0) == (b < 0):  # same sign
                    ang = ang+np.pi/2
                return ang
            else:
                ang = np.pi/2 + .5 * invcot((a-c)/(2*b))
                if (a < 0) != (b < 0):
                    ang=ang-np.pi/2
                return ang

    coords = [[], []]
    img[mask] = 0
    if interactive:
        figure1 = plt.figure()
        ax = figure1.add_subplot(111)
        plt.imshow(i)
        plt.text(x=-50, y=-20, s="Click to add points. Click again to remove points. You must have 5 points")

        def add_point(event):
            ix, iy = event.xdata, event.ydata
            x_diff, y_diff = np.abs(np.subtract(coords[0], ix)), np.abs(np.subtract(coords[1], iy))
            truth = list(np.multiply(x_diff < 10, y_diff < 10))
            if not len(x_diff):
                coords[0].append(ix)
                coords[1].append(iy)
            elif not any(truth):
                coords[0].append(ix)
                coords[1].append(iy)
            else:
                truth = list(np.multiply(x_diff < 10, y_diff < 10)).index(True)
                coords[0].pop(truth)
                coords[1].pop(truth)
            print(coords)
            ax.clear()
            ax.imshow(img)
            ax.scatter(coords[0], coords[1], s=10, marker="o", color="crimson")
            figure1.canvas.draw()
        cid = figure1.canvas.mpl_connect('button_press_event', add_point)
        plt.show()

    else:
        coords = get_max_positions(img, num_points=num_points, radius=suspected_radius)
    a = fit_ellipse(np.array(coords[0]), np.array(coords[1]))
    center = ellipse_center(a)  # (x,y)
    lengths = ellipse_axis_length(a)
    angle = ellipse_angle_of_rotation(a)
    print("The center is:", center)
    print("The major and minor axis lengths are:", lengths)
    print("The angle of rotation is:", angle)
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    R1 = [[cos_t, sin_t, 0], [-sin_t, cos_t, 0], [0, 0, 1]]
    R2 = [[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]]
    D = [[1, 0, 0], [0, lengths[0]/lengths[1], 0], [0, 0, 1]]
    M = np.matmul(np.matmul(R1, D), R2)
    return center, M


def invcot(val):
    return (np.pi/2) - np.arctan(val)


def get_max_positions(image, num_points=None, radius=None):
    i_shape = np.shape(image)
    flattened_array = image.flatten()
    indexes = np.argsort(flattened_array)

    if isinstance(flattened_array, np.ma.masked_array):
        indexes = indexes[flattened_array.mask[indexes] == False]
    if radius is not None:
        center = [np.floor_divide(np.mean(indexes[-num_points:]), i_shape[1]),
                  np.remainder(np.mean(indexes[-num_points:]), i_shape[1])]
        print(center)
    # take top 5000 points make sure exclude zero beam
    cords = [np.floor_divide(indexes[-num_points:], i_shape[1]),
             np.remainder(indexes[-num_points:], i_shape[1])]  # [x axis (row),y axis (col)]
    return cords