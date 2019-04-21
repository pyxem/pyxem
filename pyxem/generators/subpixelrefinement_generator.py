# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

"""
Generating subpixel resolution on diffraction vectors.
"""

import numpy as np

from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import peaks_as_gvectors
from pyxem.utils.subpixel_refinements_utils import _conventional_xc
from pyxem.utils.subpixel_refinements_utils import get_experimental_square
from pyxem.utils.subpixel_refinements_utils import get_simulated_disc


class SubpixelrefinementGenerator():
    """Generates subpixel refinement of DiffractionVectors.

    Parameters
    ----------
    dp : ElectronDiffraction
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

        def _floor(vectors, calibration, center):
            if vectors.shape == (1,) and vectors.dtype == np.object:
                vectors = vectors[0]
            return np.floor((vectors.astype(np.float64) / calibration) + center).astype(np.int)

        if isinstance(vectors, DiffractionVectors):
            if vectors.axes_manager.navigation_shape != dp.axes_manager.navigation_shape:
                raise ValueError('Vectors with shape {} must have the same navigation shape '
                                 'as the diffraction patterns which has shape {}.'.format(
                                     vectors.axes_manager.navigation_shape, dp.axes_manager.navigation_shape))
            self.vector_pixels = vectors.map(_floor,
                                             calibration=self.calibration,
                                             center=self.center,
                                             inplace=False)
        else:
            self.vector_pixels = _floor(vectors, self.calibration, self.center)

        if isinstance(self.vector_pixels,DiffractionVectors):
            if np.any(self.vector_pixels.data > (np.max(dp.data.shape) - 1)) or (np.any(self.vector_pixels.data < 0)):
                raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')
        elif isinstance(self.vector_pixels,np.ndarray):
            if np.any((self.vector_pixels > np.max(dp.data.shape) - 1)) or (np.any(self.vector_pixels < 0)):
                raise ValueError('Some of your vectors do not lie within your diffraction pattern, check your calibration')

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
        def _conventional_xc_map(dp, vectors, sim_disc, upsample_factor, center, calibration):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                expt_disc = get_experimental_square(dp, vector, square_size)
                shifts[i] = _conventional_xc(expt_disc, sim_disc, upsample_factor)
            return (((vectors + shifts) - center) * calibration)

        sim_disc = get_simulated_disc(square_size, disc_radius)
        self.vectors_out = DiffractionVectors(
            self.dp.map(_conventional_xc_map,
                        vectors=self.vector_pixels,
                        sim_disc=sim_disc,
                        upsample_factor=upsample_factor,
                        center=self.center,
                        calibration=self.calibration,
                        inplace=False))
        self.vectors_out.axes_manager.set_signal_dimension(0)
        self.last_method = "conventional_xc"
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
            z_adpt = np.copy(get_experimental_square(z, vector=vector, square_size=square_size))
            z_adpt[:, 0] = 0
            z_adpt[0, :] = 0
            return z_adpt

        def _center_of_mass_map(dp, vectors, square_size, center, calibration):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                expt_disc = _com_experimental_square(dp, vector, square_size)
                shifts[i] = [a - square_size / 2 for a in _center_of_mass_hs(expt_disc)]
            return ((vectors + shifts) - center) * calibration

        self.vectors_out = DiffractionVectors(
            self.dp.map(_center_of_mass_map,
                        vectors=self.vector_pixels,
                        square_size=square_size,
                        center=self.center,
                        calibration=self.calibration,
                        inplace=False))
        self.vectors_out.axes_manager.set_signal_dimension(0)

        self.last_method = "center_of_mass_method"
        return self.vectors_out

    def local_gaussian_method(self,square_size):
        """ Refinement based on the mathematics of a local maxima on a
        continious region, using the (discrete) maxima pixel as a starting point.
        See Notes.

        Parameters
        ----------
        square_size : int
            Length (in pixels) of one side of a square the contains the peak to
            be refined.

        Returns
        -------
        vector_out : DiffractionVectors
            DiffractionVectors containing the refined vectors in calibrated
            units with the same navigation shape as the diffraction patterns.

        Notes
        -----
        This method works by first locating the maximum intenisty value within teh square.
        The four adjacent pixels are then considered and used to form two independant
        quadratic equations. Solving these gives the x_center and y_center coordinates,
        which are then returned.
        """

        def _new_lg_idea(z):
                """ Internal function providing the algebra for the local_gaussian_method,
                see docstring of that function for details

                Parameters
                ----------
                z : np.array
                    subsquare containing the peak to be localised

                Returns
                -------
                (x,y) : tuple
                    Containing subpixel resolved values for the center
                """
                si = np.unravel_index(np.argmax(z),z.shape)
                z_ref = z[si[0]-1:si[0]+2,si[1]-1:si[1]+2]
                if z_ref.shape != (3,3):
                    raise ValueError("The local maxima needs to have 4 adjacent pixels")
                M = z_ref[1,1]
                LX,RX = z_ref[1,0],z_ref[1,2]
                UY,DY = z_ref[0,1],z_ref[2,1]
                x_ans = 0.5 * (LX-RX) / (LX + RX - 2*M)
                y_ans = 0.5 * (UY-DY) / (UY + DY - 2*M)
                return (si[1]+x_ans,si[0]+y_ans)

        def _lg_map(dp, vectors,square_size,center,calibration):
            shifts = np.zeros_like(vectors, dtype=np.float64)
            for i, vector in enumerate(vectors):
                expt_disc = get_experimental_square(dp, vector, square_size)
                shifts[i] = _new_lg_idea(expt_disc)
            return (((vectors + shifts) - center) * calibration)

        self.vectors_out = DiffractionVectors(
        self.dp.map(_lg_map,
                    vectors=self.vector_pixels,
                    square_size = square_size,
                    center=self.center,
                    calibration=self.calibration,
                    inplace=False))

        self.vectors_out.axes_manager.set_signal_dimension(0)
        self.last_method = "lg_method"
        return self.vectors_out
