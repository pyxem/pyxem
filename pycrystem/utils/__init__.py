# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import RectBivariateSpline

import pycrystem.utils.strain_utils


def correlate(image, pattern, include_direct_beam=False, sim_threshold=1e-5,
              **kwargs):
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
    sim_threshold : float
        The threshold simulation intensity to consider for correlation
    **kwargs
        Arguments to pass to scipy.interpolate.RectBivariateSpline
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
    half_shape = tuple(i // 2 for i in shape)
    x = np.arange(shape[0], dtype='float') - half_shape[0]
    y = np.arange(shape[1], dtype='float') - half_shape[1]
    for ar, i in zip([x, y], shape):
        if not i % 2:
            ar += 0.5
    x = x * pattern.calibration[0]
    y = y * pattern.calibration[1]

    pixel_coordinates = pattern.calibrated_coordinates.astype(int)[
        :, :2] + half_shape
    in_bounds = np.product((pixel_coordinates > 0) *
                           (pixel_coordinates < shape[0]), axis=1).astype(bool)
    pattern_intensities = pattern.intensities
    large_intensities = pattern_intensities > sim_threshold
    mask = np.logical_and(in_bounds, large_intensities)

    ip = RectBivariateSpline(x, y, image.T, **kwargs)
    image_intensities = ip.ev(pattern.coordinates[:, 0][mask],
                              pattern.coordinates[:, 1][mask])
    pattern_intensities = pattern_intensities[mask]
    return np.nan_to_num(_correlate(image_intensities, pattern_intensities))


def correlate_component(image, pattern):
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
