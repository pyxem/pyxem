# -*- coding: utf-8 -*-

import numpy as np


def correlate(image, pattern, scale=1., offset=(0., 0.),
              axes_manager=None):
    """The correlation between a diffraction pattern and a simulation.

    Calculated using
        .. math::
            \frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image : {:class:`ElectronDiffraction`, :class:`ndarray`}
        Either a single electron diffraction signal (should be appropriately scaled
        and centered) or a 1D or 2D numpy array.
    pattern : :class:`DiffractionSimulation`
        The pattern to compare to.
    scale : float
        A scale to fine-tune correlation.
    offset : :obj:tuple of :obj:float
        A centre offset to fine-tune correlation.
    axes_manager : :class:`AxesManager`
        If image is an array, the appropriate AxesManager to get axes, scaling
        and shape information

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
    if axes_manager is None:
        if isinstance(image, np.ndarray):
            raise ValueError("No scaling information given")
        else:
            axes_manager = image.axes_manager
            image = image.data
    x_axis = axes_manager.signal_axes[0]
    y_axis = axes_manager.signal_axes[1]

    # Ensure correct data shape
    shape = image.shape
    if len(shape) not in [1, 2]:
        raise ValueError("Only support 2D and 1D (i.e. 'raveled 2D') arrays")
    else:
        if len(shape) == 1:
            image = image.reshape(axes_manager.signal_shape[::-1])

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
    image_intensities = image[x[condition], y[condition]]
    pattern_intensities = pattern.intensities[condition]
    return _correlate(image_intensities, pattern_intensities)


def _correlate(intensities_1, intensities_2):
    return np.dot(intensities_1, intensities_2) / (
        np.sqrt(np.dot(intensities_1, intensities_1)) *
        np.sqrt(np.dot(intensities_2, intensities_2))
    )
