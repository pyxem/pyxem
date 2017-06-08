# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, \
    FloatingSubplot
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from scipy.interpolate import griddata
from pymatgen.transformations.standard_transformations \
    import RotationTransformation
from transforms3d.euler import euler2axangle

# from pycrystem import Structure, ElectronDiffractionCalculator
from pycrystem.utils import correlate

from ipywidgets import interact


def symmetry_axes(figure, theta=(0, np.pi / 4), phi=(-np.pi / 4, np.pi / 4)):
    theta_min, theta_max = theta[0], theta[1]
    phi_min, phi_max = phi[0], phi[1]
    transform = Affine2D().translate(0, 0) + PolarAxes.PolarTransform()
    grid_helper = GridHelperCurveLinear(
        transform,
        (phi_max, phi_min, theta_max, theta_min),
    )
    ax = FloatingSubplot(figure, 111, grid_helper=grid_helper)
    figure.add_subplot(ax)
    aux_ax = ax.get_aux_axes(transform)
    aux_ax.patch = ax.patch
    ax.patch.zorder = 0.9
    return aux_ax


def plot_correlation_map(
        angles,
        correlations,
        levels=30,
        interpolation_method='cubic',
        resolution=0.001,
        theta=(0, np.pi / 4),
        phi=(-np.pi / 4, np.pi / 4)
):
    xy = angles
    z = correlations
    z = z - z.min()
    z = 1 - z / z.max()

    grid_x, grid_y = np.mgrid[phi[0]:phi[1]:resolution,
                     theta[0]:theta[1]:resolution]
    grid = griddata(xy, z, (grid_x, grid_y), method=interpolation_method)
    fig = plt.figure()
    ax = symmetry_axes(fig, theta=theta, phi=phi)
    ax.contourf(grid_x, grid_y, grid, levels)
    return ax


def manual_orientation(
        data,  #: np.ndarray,
        structure,  #: Structure,
        calculator,  #: ElectronDiffractionCalculator,
        ax=None,
):
    if ax is None:
        ax = plt.figure().add_subplot(111)
    dimension = data.shape[0] / 2
    extent = [-dimension, dimension] * 2
    ax.imshow(data, extent=extent, interpolation='none', origin='lower')
    text = plt.text(dimension, dimension, "Loading...")
    p = plt.scatter([0, ], [0, ], s=0)

    def plot(alpha=0., beta=0., gamma=0., calibration=1., reciprocal_radius=1.0):
        calibration /= 100
        orientation = euler2axangle(alpha, beta, gamma, 'rzyz')
        rotation = RotationTransformation(orientation[0], orientation[1],
                                          angle_in_radians=True).apply_transformation(
            structure)
        electron_diffraction = calculator.calculate_ed_data(rotation, reciprocal_radius)
        electron_diffraction.calibration = calibration
        nonlocal p
        p.remove()
        p = plt.scatter(
            electron_diffraction.calibrated_coordinates[:, 0],
            electron_diffraction.calibrated_coordinates[:, 1],
            s=electron_diffraction.intensities,
            facecolors='none',
            edgecolors='r'
        )
        text.set_text('\n'.join([str(correlate(data, electron_diffraction)), str(calibration)]))
        ax.set_xlim(-dimension, dimension)
        ax.set_ylim(-dimension, dimension)
        plt.show()

    interact(plot, alpha=(-np.pi, np.pi, 0.01), beta=(-np.pi, np.pi, 0.01),
             gamma=(-np.pi, np.pi, 0.01), calibration=(1e-2, 1e1, 1e-2),
             reciprocal_radius=(1e-1, 5., 1e-1))
