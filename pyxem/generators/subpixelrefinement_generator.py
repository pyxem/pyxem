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
from skimage.registration import phase_cross_correlation
from skimage import draw

from pyxem.signals import DiffractionVectors


def get_experimental_square(z, vector, square_size):
    """Defines a square region around a given diffraction vector and returns.

    Parameters
    ----------
    z : np.array()
        Single diffraction pattern
    vector : np.array()
        Single vector in pixels (int) [x,y] with top left as [0,0]
    square_size : int
        The length of one side of the bounding square (must be even)

    Returns
    -------
    square : np.array()
        Of size (L,L) where L = square_size

    """
    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    cx, cy, half_ss = vector[0], vector[1], int(square_size / 2)
    # select square with correct x,y see PR for details
    _z = z[cy - half_ss : cy + half_ss, cx - half_ss : cx + half_ss]
    return _z


def get_simulated_disc(square_size, disc_radius):
    """Create a uniform disc for correlating with the experimental square.

    Parameters
    ----------
    square size : int
        (even) - size of the bounding box
    disc_radius : int
        radius of the disc

    Returns
    -------
    arr: np.array()
        Upsampled copy of the simulated disc as a numpy array

    """

    if square_size % 2 != 0:
        raise ValueError("'square_size' must be an even number")

    ss = int(square_size)
    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(
        int(ss / 2), int(ss / 2), radius=disc_radius, shape=arr.shape
    )  # is the thin disc a good idea
    arr[rr, cc] = 1
    return arr


def _get_pixel_vectors(dp, vectors, calibration, center):
    """Get the pixel coordinates for the given diffraction
    pattern and vectors.

    Parameters
    ----------
    dp: :obj:`pyxem.signals.ElectronDiffraction2D`
        Instance of ElectronDiffraction2D
    vectors : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors`
        List of diffraction vectors
    calibration : [float, float]
        Calibration values
    center : float, float
        Image origin in pixel coordinates

    Returns
    -------
    vector_pixels : :obj:`pyxem.signals.diffraction_vectors.DiffractionVectors`
        Pixel coordinates for given diffraction pattern and vectors.
    """

    def _floor(vectors, calibration, center):
        if vectors.shape == (1,) and vectors.dtype == object:
            vectors = vectors[0]
        return np.floor((vectors.astype(np.float64) / calibration) + center).astype(
            np.int
        )

    if isinstance(vectors, DiffractionVectors):
        if vectors.axes_manager.navigation_shape != dp.axes_manager.navigation_shape:
            raise ValueError(
                "Vectors with shape {} must have the same navigation shape "
                "as the diffraction patterns which has shape {}.".format(
                    vectors.axes_manager.navigation_shape,
                    dp.axes_manager.navigation_shape,
                )
            )
        vector_pixels = vectors.map(
            _floor, calibration=calibration, center=center, inplace=False
        )
    else:
        vector_pixels = _floor(vectors, calibration, center)

    if isinstance(vector_pixels, DiffractionVectors):
        if np.any(vector_pixels.data > (np.max(dp.data.shape) - 1)) or (
            np.any(vector_pixels.data < 0)
        ):
            raise ValueError(
                "Some of your vectors do not lie within your diffraction pattern, check your calibration"
            )
    elif isinstance(vector_pixels, np.ndarray):
        if np.any((vector_pixels > np.max(dp.data.shape) - 1)) or (
            np.any(vector_pixels < 0)
        ):
            raise ValueError(
                "Some of your vectors do not lie within your diffraction pattern, check your calibration"
            )

    return vector_pixels


def _conventional_xc(exp_disc, sim_disc, upsample_factor):
    """Takes two images of disc and finds the shift between them using
    conventional (phase) cross correlation.

    Parameters
    ----------
    exp_disc : np.array()
        A numpy array of the "experimental" disc
    sim_disc : np.array()
        A numpy array of the disc used as a template
    upsample_factor: int (must be even)
        Factor to upsample by, reciprocal of the subpixel resolution
        (eg 10 ==> 1/10th of a pixel)

    Returns
    -------
    shifts
        Pixel shifts required to register the two images

    """

    shifts, error, _ = phase_cross_correlation(
        exp_disc, sim_disc, upsample_factor=upsample_factor
    )
    shifts = np.flip(shifts)  # to comply with hyperspy conventions - see issue#490
    return shifts


class SubpixelrefinementGenerator:
    """Generates subpixel refinement of DiffractionVectors.

    Parameters
    ----------
    dp : ElectronDiffraction2D
        The electron diffraction patterns to be refined
    vectors : DiffractionVectors | ndarray
        Vectors (in calibrated units) to the locations of the spots to be
        refined. If given as DiffractionVectors, it must have the same
        navigation shape as the electron diffraction patterns. If an ndarray,
        the same set of vectors is mapped over all electron diffraction
        patterns.

    References
    ----------
    [1] Pekin et al. Ultramicroscopy 176 (2017) 170-176

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

    def conventional_xc(self, square_size, disc_radius, upsample_factor):
        """Refines the peaks using (phase) cross correlation.

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to
            be refined.
        disc_radius:  int
            Radius (in pixels) of the discs that you seek to refine
        upsample_factor: int
            Factor by which to upsample the patterns

        Returns
        -------
        vector_out: DiffractionVectors
            DiffractionVectors containing the refined vectors in calibrated
            units with the same navigation shape as the diffraction patterns.

        """

        def _conventional_xc_map(
            dp, vectors, sim_disc, upsample_factor, center, calibration
        ):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                expt_disc = get_experimental_square(dp, vector, square_size)
                shifts[i] = _conventional_xc(expt_disc, sim_disc, upsample_factor)
            return ((vectors + shifts) - center) * calibration

        sim_disc = get_simulated_disc(square_size, disc_radius)
        self.vectors_out = self.dp.map(
            _conventional_xc_map,
            vectors=self.vector_pixels,
            sim_disc=sim_disc,
            upsample_factor=upsample_factor,
            center=self.center,
            calibration=self.calibration,
            inplace=False,
        )
        self.vectors_out.set_signal_type("diffraction_vectors")

        self.last_method = "conventional_xc"
        return self.vectors_out

    def reference_xc(self, square_size, reference_dp, upsample_factor):
        """Refines the peaks using (phase) cross correlation with a reference
        diffraction image.

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to
            be refined.
        reference_dp: ndarray
            Same shape as a single diffraction image
        upsample_factor: int
            Factor by which to upsample the patterns

        Returns
        -------
        vector_out: DiffractionVectors
            DiffractionVectors containing the refined vectors in calibrated
            units with the same navigation shape as the diffraction patterns.

        """

        def _reference_xc_map(dp, vectors, upsample_factor, center, calibration):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                ref_disc = get_experimental_square(reference_dp, vector, square_size)
                expt_disc = get_experimental_square(dp, vector, square_size)
                shifts[i] = _conventional_xc(expt_disc, ref_disc, upsample_factor)
            return ((vectors + shifts) - center) * calibration

        self.vectors_out = self.dp.map(
            _reference_xc_map,
            vectors=self.vector_pixels,
            upsample_factor=upsample_factor,
            center=self.center,
            calibration=self.calibration,
            inplace=False,
        )
        self.vectors_out.set_signal_type("diffraction_vectors")

        self.last_method = "reference_xc"
        return self.vectors_out

    def center_of_mass_method(self, square_size):
        """Find the subpixel refinement of a peak by assuming it lies at the
        center of intensity.

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to
            be refined.

        Returns
        -------
        vector_out: DiffractionVectors
            DiffractionVectors containing the refined vectors in calibrated
            units with the same navigation shape as the diffraction patterns.

        """

        def _center_of_mass_hs(z):
            """Return the center of mass of an array with coordinates in the
            hyperspy convention

            Parameters
            ----------
            z : np.array

            Returns
            -------
            (x,y) : tuple of floats
                The x and y locations of the center of mass of the parsed square
            """

            s = np.sum(z)
            if s != 0:
                z *= 1 / s
            dx = np.sum(z, axis=0)
            dy = np.sum(z, axis=1)
            h, w = z.shape
            cx = np.sum(dx * np.arange(w))
            cy = np.sum(dy * np.arange(h))
            return cx, cy

        def _com_experimental_square(z, vector, square_size):
            """Wrapper for get_experimental_square that makes the non-zero
            elements symmetrical around the 'unsubpixeled' peak by zeroing a
            'spare' row and column (top and left).

            Parameters
            ----------
            z : np.array

            vector : np.array([x,y])

            square_size : int (even)

            Returns
            -------
            z_adpt : np.array
                z, but with row and column zero set to 0
            """
            # Copy to make sure we don't change the dp
            z_adpt = np.copy(
                get_experimental_square(z, vector=vector, square_size=square_size)
            )
            z_adpt[:, 0] = 0
            z_adpt[0, :] = 0
            return z_adpt

        def _center_of_mass_map(dp, vectors, square_size, center, calibration):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                expt_disc = _com_experimental_square(dp, vector, square_size)
                shifts[i] = [a - square_size / 2 for a in _center_of_mass_hs(expt_disc)]
            return ((vectors + shifts) - center) * calibration

        self.vectors_out = self.dp.map(
            _center_of_mass_map,
            vectors=self.vector_pixels,
            square_size=square_size,
            center=self.center,
            calibration=self.calibration,
            inplace=False,
        )
        self.vectors_out.set_signal_type("diffraction_vectors")

        self.last_method = "center_of_mass_method"
        return self.vectors_out

    def local_gaussian_method(self, square_size):
        """Removed in v0.13, please install a version prior to v.0.13 to use."""
        raise NotImplementedError(
            "This functionality was removed in v.0.13.0, please use another method"
        )
