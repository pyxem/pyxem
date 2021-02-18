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

"""Generating subpixel resolution on diffraction vectors."""

import numpy as np
from skimage import morphology
from skimage.measure import label
from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass

from hyperspy.signals import BaseSignal

from pyxem.generators.subpixelrefinement_generator import _get_pixel_vectors
from pyxem.signals import DiffractionVectors


def _get_intensities(z, vectors, radius=1):
    """Basic intensity integration routine, takes the maximum value at the
    given vector positions with the number of pixels given by `radius`.

    Parameters
    ----------
    vectors : DiffractionVectors
        Vectors to the locations of the spots to be
        integrated.
    radius: int,
        Number of pixels within which to find the largest maximum

    Returns
    -------
    intensities : np.array
        List of extracted intensities
    """
    i, j = np.array(vectors.data).astype(int).T

    if radius > 1:
        footprint = morphology.disk(radius)
        filtered = ndi.maximum_filter(z, footprint=footprint)
        intensities = filtered[j, i].reshape(-1, 1)  # note that the indices are flipped
    else:
        intensities = z[j, i].reshape(-1, 1)  # note that the indices are flipped

    return np.array(intensities)


def _take_ragged(z, indices, _axis=None, out=None, mode="raise"):
    """Like `np.take` for ragged arrays, see `np.take` for documentation."""
    return np.take(z[0], indices, axis=_axis, out=out, mode=mode)


def _get_largest_connected_region(segmentation):
    """Take a binary segmentation image and return the largest connected area."""
    labels = label(segmentation)
    largest = np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return (labels == largest).astype(int)


def _get_intensities_summation_method(
    z,
    vectors,
    box_inner: int = 7,
    box_outer: int = 10,
    n_min: int = 5,
    n_max: int = None,
    snr_thresh=3.0,
    verbose: bool = False,
):
    """Integrate reflections using the summation method.

    Two boxes are defined, the inner box is used to define the
    integration area. The outer box is used to calculate the
    average signal-to-noise ratio (SNR).

    All pixels with a large enough SNR are considered to be signal. The largest region
    of connected signal pixels are summed to calculate the reflection intensity.

    Parameters
    ----------
    vectors : DiffractionVectors
        Vectors to the locations of the spots to be
        integrated.
    box_inner : int
        Defines the radius (size) of the inner box, which must be larger than the reflection.
        The total box size is 2*box_inner
    box_outer : int
        Defines radius (size) of the outer box. The total box size is 2*box_inner
        The border between the inner and outer box is considered background
        and used to calculate the (SNR) for each pixel: SNR = (I - <I>/std(I_bkg)).
    snr_thresh : float
        Minimum signal-to-noise for a pixel to be considered as `signal`.
    n_min: int
        If the number of SNR pixels in the inner box < n_min, the reflection is discared
    n_max:
        If the number of SNR pixels in the inner box >= n_max, the reflection is discareded
        Defaults to the inner box size (`box_inner**2`.
    verbose : bool
        Print statistics for every reflection (for debugging)

    Returns
    -------
    peaks : np.array
        Array with 4 columns: X-position, Y-position, intensity, reflection SNR

    Notes
    -----
    Implementation based on Barty et al, J. Appl. Cryst. (2014). 47, 1118-1131
    Lesli, Acta Cryst. (2006). D62, 48-57

    """
    if not n_max:  # pragma: no cover
        n_max = box_inner ** 2

    peaks = []

    for i, j in vectors:
        box = z[j - box_inner : j + box_inner, i - box_inner : i + box_inner].copy()

        bkg = np.hstack(
            [
                z[j - box_outer : j + box_outer, i - box_outer : i - box_inner].ravel(),
                z[j - box_outer : j + box_outer, i + box_inner : i + box_outer].ravel(),
                z[j - box_outer : j - box_inner, i - box_inner : i + box_inner].ravel(),
                z[j + box_inner : j + box_outer, i - box_inner : i + box_inner].ravel(),
            ]
        )

        bkg_mean = bkg.mean()
        bkg_std = bkg.std()
        box_snr = (box - bkg_mean) / bkg_std

        # get mask for signal (I > SNR)
        signal_mask = _get_largest_connected_region(box_snr > snr_thresh)

        n_pix = signal_mask.sum()
        signal = (box - bkg_mean) * signal_mask
        inty = signal.sum()
        snr = (inty / n_pix) / bkg_std
        sigma = inty / snr

        # calculate center of mass
        com_X, com_Y = center_of_mass(box, labels=signal_mask, index=1)
        dX = com_X - box_inner
        dY = com_Y - box_inner

        X = i + dX
        Y = j + dY

        if verbose:  # pragma: no cover
            print(
                f"\nMean(I): {bkg_mean:.2f} | Std(I): {bkg_std:.2f} | n_pix: {n_pix} \n"
                f"I: {inty:.2f} | Sigma(I): {sigma:.2f} | SNR(I): {snr:.2f} | I/pix: {inty/n_pix:.2f} \n"
                f"i: {i:.2f} | j: {j:.2f} | dX: {dX:.2f} | dY: {dY:.2f} | X: {X:.2f} | Y: {Y:.2f} "
            )
            # for debugging purposes
            import matploltib.pyplot as plt

            plt.imshow(signal)
            plt.plot(dY + box_inner, dX + box_inner, "r+")  # center_of_mass
            plt.plot(box_inner, box_inner, "g+")  # input
            plt.show()

        if n_pix > n_max:  # pragma: no cover
            continue
        if n_pix < n_min:  # pragma: no cover
            continue

        # for some reason X/Y are reversed here
        peaks.append([Y, X, inty, sigma])

    peaks = np.array(peaks)

    return np.array(peaks)


class IntegrationGenerator:
    """Integrates reflections at the given vector positions.

    Parameters
    ----------
    dp : ElectronDiffraction2D
        The electron diffraction patterns to be refined
    vectors : DiffractionVectors | ndarray
        Vectors (in calibrated units) to the locations of the spots to be
        integrated. If given as DiffractionVectors, it must have the same
        navigation shape as the electron diffraction patterns. If an ndarray,
        the same set of vectors is mapped over all electron diffraction
        patterns.
    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        self.last_method = None
        sig_ax = dp.axes_manager.signal_axes
        self.calibration = [sig_ax[0].scale, sig_ax[1].scale]
        self.center = [sig_ax[0].size / 2, sig_ax[1].size / 2]

        self.vector_pixels = _get_pixel_vectors(
            dp, vectors, calibration=self.calibration, center=self.center
        )

    def extract_intensities(self, radius: int = 1):
        """Basic intensity integration routine, takes the maximum value at the
        given vector positions with the number of pixels given by `radius`.

        Parameters
        ----------
        radius: int,
            Number of pixels within which to find the largest maximum

        Returns
        -------
        intensities : :obj:`hyperspy.signals.BaseSignal`
            List of extracted intensities
        """
        intensities = self.dp.map(
            _get_intensities, vectors=self.vector_pixels, radius=radius, inplace=False
        )

        intensities = BaseSignal(intensities)
        intensities.axes_manager.set_signal_dimension(0)

        return intensities

    def extract_intensities_summation_method(
        self,
        box_inner: int = 7,
        box_outer: int = 10,
        n_min: int = 5,
        n_max: int = 1000,
        snr_thresh: float = 3.0,
    ):
        """Integrate reflections using the summation method. Two boxes are defined,
        the inner box is used to define the integration area. The outer box is used
        to calculate the average signal-to-noise ratio (SNR).
        All pixels with a large enough SNR are considered to be signal. The largest region
        of connected signal pixels are summed to calculate the reflection intensity. The
        diffraction vectors are calculated as the center of mass of the signal pixels.

        Parameters
        ----------
        box_inner : int
            Defines the size of the inner box, which must be larger than the reflection.
        box_outer : int
            Defines the size of the outer box. The border between the inner and outer
            box is considered background and used to calculate the (SNR) for each
            pixel: SNR = (I - <I>/std(I_bkg)).
        snr_thresh : float
            Minimum signal-to-noise for a pixel to be considered as `signal`.
        n_min: int
            If the number of SNR pixels in the inner box < n_min, the reflection is discared
        n_max:
            If the number of SNR pixels in the inner box > n_max, the reflection is discareded
        verbose : bool
            Print statistics for every reflection (for debugging)

        Returns
        -------
        vectors : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors`
            DiffractionVectors with optimized coordinates, where the attributes
            vectors.intensities -> `I`, vectors.sigma -> `sigma(I)`, and
            vectors.snr -> `I / sigma(I)`

        Notes
        -----
        Implementation based on Barty et al, J. Appl. Cryst. (2014). 47, 1118-1131
                                Lesli, Acta Cryst. (2006). D62, 48-57
        """
        result = self.dp.map(
            _get_intensities_summation_method,
            vectors=self.vector_pixels,
            box_inner=box_inner,
            box_outer=box_outer,
            n_min=n_min,
            n_max=n_max,
            snr_thresh=snr_thresh,
            inplace=False,
            ragged=True,
        )

        peaks = result.map(
            _take_ragged, indices=[0, 1], _axis=1, inplace=False, ragged=True
        )
        intensities = result.map(
            _take_ragged, indices=2, _axis=1, inplace=False, ragged=True
        )
        sigma = result.map(_take_ragged, indices=3, _axis=1, inplace=False, ragged=True)

        vectors = DiffractionVectors.from_peaks(
            peaks, calibration=self.calibration, center=self.center
        )
        vectors.intensities = intensities
        vectors.sigma = sigma
        vectors.snr = intensities / sigma

        return vectors
