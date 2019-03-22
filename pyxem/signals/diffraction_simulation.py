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

import matplotlib.pyplot as plt
import numpy as np

from pyxem.signals.electron_diffraction import ElectronDiffraction


class DiffractionSimulation:
    """Holds the result of a kinematic diffraction pattern simulation.

    Parameters
    ----------
    coordinates : array-like, shape [n_points, 2]
        The x-y coordinates of points in reciprocal space.
    indices : array-like, shape [n_points, 3]
        The indices of the reciprocal lattice points that intersect the
        Ewald sphere.
    intensities : array-like, shape [n_points, ]
        The intensity of the reciprocal lattice points.
    calibration : float or tuple of float, optional
        The x- and y-scales of the pattern, with respect to the original
        reciprocal angstrom coordinates.
    offset : tuple of float, optional
        The x-y offset of the pattern in reciprocal angstroms. Defaults to
        zero in each direction.
    """

    def __init__(self, coordinates=None, indices=None, intensities=None,
                 calibration=1., offset=(0., 0.), with_direct_beam=False):
        """Initializes the DiffractionSimulation object with data values for
        the coordinates, indices, intensities, calibration and offset.
        """
        self._coordinates = None
        self.coordinates = coordinates
        self.indices = indices
        self._intensities = None
        self.intensities = intensities
        self._calibration = (1., 1.)
        self.calibration = calibration
        self.offset = offset
        self.with_direct_beam = with_direct_beam

    @property
    def calibrated_coordinates(self):
        """ndarray : Coordinates converted into pixel space."""
        coordinates = np.copy(self.coordinates)
        coordinates[:, 0] += self.offset[0]
        coordinates[:, 1] += self.offset[1]
        coordinates[:, 0] /= self.calibration[0]
        coordinates[:, 1] /= self.calibration[1]
        return coordinates

    @property
    def calibration(self):
        """tuple of float : The x- and y-scales of the pattern, with respect to
        the original reciprocal angstrom coordinates."""
        return self._calibration

    @calibration.setter
    def calibration(self, calibration):
        if np.all(np.equal(calibration, 0)):
            raise ValueError("`calibration` cannot be zero.")
        if isinstance(calibration, float) or isinstance(calibration, int):
            self._calibration = (calibration, calibration)
        elif len(calibration) == 2:
            self._calibration = calibration
        else:
            raise ValueError("`calibration` must be a float or length-2"
                             "tuple of floats.")

    @property
    def direct_beam_mask(self):
        """ndarray : If `with_direct_beam` is True, returns a True array for all
        points. If `with_direct_beam` is False, returns a True array with False
        in the position of the direct beam."""
        if self.with_direct_beam:
            return np.ones_like(self._intensities, dtype=bool)
        else:
            return np.any(self._coordinates, axis=1)

    @property
    def coordinates(self):
        """ndarray : The coordinates of all unmasked points."""
        if self._coordinates is None:
            return None
        return self._coordinates[self.direct_beam_mask]

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def intensities(self):
        """ndarray : The intensities of all unmasked points."""
        if self._intensities is None:
            return None
        return self._intensities[self.direct_beam_mask]

    @intensities.setter
    def intensities(self, intensities):
        self._intensities = intensities

    def as_signal(self, size, sigma, max_r):
        """Returns the diffraction data as an ElectronDiffraction signal with
        two-dimensional Gaussians representing each diffracted peak. Should only
        be used for qualitative work.

        Parameters
        ----------
        size : int
            Side length (in pixels) for the signal to be simulated.
        sigma : float
            Standard deviation of the Gaussian function to be plotted.
        max_r : float
            Half the side length in reciprocal Angstroms. Defines the signal's
            calibration

        Returns
        -------

        dp : ElectronDiffraction
            Simulated electron diffraction pattern.

        """
        from skimage.filters import gaussian as point_spread

        l, delta_l = np.linspace(-max_r, max_r, size, retstep=True)

        mask_for_max_r = np.logical_and(np.abs(self.coordinates[:, 0]) < max_r,
                                        np.abs(self.coordinates[:, 1]) < max_r)

        coords = self.coordinates[mask_for_max_r]
        inten = self.intensities[mask_for_max_r]

        dp_dat = np.zeros([size, size])
        x, y = (coords)[:, 0], (coords)[:, 1]
        if len(x) > 0:  # avoiding problems in the peakless case
            num = np.digitize(x, l, right=True), np.digitize(y, l, right=True)
            dp_dat[num] = inten
            # sigma in terms of pixels. transpose for Hyperspy
            dp_dat = point_spread(dp_dat, sigma=sigma / delta_l).T
            dp_dat = dp_dat / np.max(dp_dat)

        dp = ElectronDiffraction(dp_dat)
        dp.set_diffraction_calibration(2 * max_r / size)

        return dp


class ProfileSimulation:
    """Holds the result of a given kinematic simulation of a diffraction profile

    Parameters
    ----------
    magnitudes : array-like, shape [n_peaks, 1]
        Magnitudes of scattering vectors.
    intensities : array-like, shape [n_peaks, 1]
        The kinematic intensity of the diffraction peaks.
    hkls: [{(h, k, l): mult}] {(h, k, l): mult} is a dict of Miller
        indices for all diffracted lattice facets contributing to each
        intensity.
    """

    def __init__(self, magnitudes, intensities, hkls):
        """Initializes the ProfileSimulation object with data values for the
        magnitudes, intensities, and hkls.
        """
        self.magnitudes = magnitudes
        self.intensities = intensities
        self.hkls = hkls

    def get_plot(self, g_max, annotate_peaks=True,
                 with_labels=True, fontsize=12):
        """Plots the diffraction profile simulation.

        Parameters
        ----------
        g_max : float
            Maximum g-vector magnitude to plot.
        annotate_peaks : boolean
            If True, peaks are annotaed with hkl information.
        with_labels : boolean
            If True, xlabels and ylabels are added to the plot.
        fontsize : integer
            Fontsize for peak labels.
        """
        ax = plt.gca()
        for g, i, hkls in zip(self.magnitudes, self.intensities, self.hkls):
            if g <= g_max:
                label = ", ".join([str(hkl) for hkl in hkls.keys()])
                ax.plot([g, g], [0, i], color='k',
                        linewidth=3, label=label)
                if annotate_peaks:
                    ax.annotate(label, xy=[g, i],
                                xytext=[g, i], fontsize=fontsize)

            if with_labels:
                ax.set_xlabel("A ($^{-1}$)")
                ax.set_ylabel("Intensities (scaled)")

        return plt
