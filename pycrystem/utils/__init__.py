# -*- coding: utf-8 -*-
import numpy as np

import pycrystem.utils.strain_utils


def correlate(image, pattern, include_direct_beam=False):
    """The correlation between a diffraction pattern and a simulation.
    Calculated using
        .. math::
            \frac{\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\sqrt{\sum_{j=1}^m P^2(x_j, y_j)} \sqrt{\sum_{j=1}^m T^2(x_j, y_j)}}
    Parameters
    ----------
    image : :class:`np.ndarray`
        A single electron diffraction signal. Should be appropriately scaled
        and centered.
    pattern : :class:`DiffractionSimulation`
        The pattern to compare to.
    Returns
    -------
    float
        The correlation coefficient.
    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
        template matching,” vol. 50, no. 1, pp. 87–99, 2005.
    """
    shape = image.shape
    half_shape = tuple(int(i / 2) for i in shape)
    pixel_coordinates = pattern.calibrated_coordinates.astype(int)[:, :2] + half_shape
    in_bounds = np.product((pixel_coordinates > 0) * (pixel_coordinates < shape[0]), axis=1).astype(bool)
    image_intensities = image.T[pixel_coordinates[:, 0][in_bounds], pixel_coordinates[:, 1][in_bounds]]
    pattern_intensities = pattern.intensities[in_bounds]
    return np.nan_to_num(_correlate(image_intensities, pattern_intensities))


def correlate_component(image, pattern):
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
    image_intensities = np.array(
        [image.isig[c[0], c[1]].data for c in pattern.coordinates]
    ).flatten()
    pattern_intensities = pattern.intensities
    return _correlate(image_intensities, pattern_intensities)


def _correlate(intensities_1, intensities_2):
    return np.dot(intensities_1, intensities_2) / (
        np.sqrt(np.dot(intensities_1, intensities_1)) *
        np.sqrt(np.dot(intensities_2, intensities_2))
    )
