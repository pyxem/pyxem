# -*- coding: utf-8 -*-

import numpy as np


def correlate(image, pattern, scale=1., offset=(0., 0.)):
    """The correlation between a diffraction pattern and a simulation.

    Calculated using
        .. math::
            \frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image : :class:`ElectronDiffraction`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern : :class:`DiffractionSimulation`
        The pattern to compare to.
    scale : float
        A scale to fine-tune correlation.
    offset : :obj:tuple of :obj:float
        A centre offset to fine-tune correlation.

    Returns
    -------
    float
        The correlation coefficient.

    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
        template matching,” vol. 50, no. 1, pp. 87–99, 2005.

    """
    # Fetch the axes
    x_axis = image.axes_manager.signal_axes[0]
    y_axis = image.axes_manager.signal_axes[1]

    # Transform the pattern into image pixel space
    x = pattern.coordinates[:, 0]
    y = pattern.coordinates[:, 1]
    x = x/x_axis.scale * scale
    y = y/y_axis.scale * scale
    x -= x_axis.offset/x_axis.scale + offset[0]
    y -= y_axis.offset/y_axis.scale + offset[1]
    x = x.astype(int)
    y = y.astype(int)

    # Constrain the positions to avoid `IndexError`s
    x_bounds = np.logical_and(0 <= x, x < x_axis.size)
    y_bounds = np.logical_and(0 <= y, y < y_axis.size)
    condition = np.logical_and(x_bounds, y_bounds)

    # Get point-by-point intensities
    image_intensities = image.data[x[condition], y[condition]]
    pattern_intensities = pattern.intensities[condition]
    return _correlate(image_intensities, pattern_intensities)


def _correlate(intensities_1, intensities_2):
    return np.dot(intensities_1, intensities_2) / (
        np.sqrt(np.dot(intensities_1, intensities_1)) *
        np.sqrt(np.dot(intensities_2, intensities_2))
    )
