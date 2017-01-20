import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, \
    FloatingSubplot
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from scipy.interpolate import griddata


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