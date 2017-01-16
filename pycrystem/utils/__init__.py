# -*- coding: utf-8 -*-
import numpy as np

import pycrystem.utils.strain_utils


def correlate(image, pattern, axes_manager, include_direct_beam=False):
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

    pixel_coordinates = pattern.calibrated_coordinates.astype(int)[:, :2] + [72, 72]
    in_bounds = np.product((pixel_coordinates > 0) * (pixel_coordinates < 144), axis=1).astype(bool)
    image_intensities = image.data.T[pixel_coordinates[:, 0][in_bounds], pixel_coordinates[:, 1][in_bounds]]
    pattern_intensities = pattern.intensities[in_bounds]
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
