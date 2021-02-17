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

"""Reduced intensity generator and associated tools."""
import numpy as np

from pyxem.components import ScatteringFitComponentXTables, ScatteringFitComponentLobato
from pyxem.utils.ri_utils import subtract_pattern, mask_from_pattern

scattering_factor_dictionary = {
    "lobato": ScatteringFitComponentLobato,
    "xtables": ScatteringFitComponentXTables,
}


class ReducedIntensityGenerator1D:
    """Generates a reduced intensity 1D profile for a specified diffraction radial profile.

    Parameters
    ----------
    signal : ElectronDiffraction1D
        An electron diffraction radial profile.
    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal
        self.cutoff = [0, signal.axes_manager.signal_axes[0].size - 1]
        self.nav_size = [
            signal.axes_manager.navigation_axes[0].size,
            signal.axes_manager.navigation_axes[1].size,
        ]
        self.sig_size = [signal.axes_manager.signal_axes[0].size]
        self.background_fit = None  # added in one of the fits below.
        self.normalisation = None

    def set_diffraction_calibration(self, calibration):
        """Defines calibration for the signal axis variable s in terms of
        A^-1 per pixel. Note that s is defined here as
        s = 2 sin(theta)/lambda = 1/d, where theta is the scattering angle,
        lambda the wavelength, and d the reciprocal spacing.

        Parameters
        ----------
        calibration: float
                    Scattering vector calibration in terms of A^-1 per pixel.
        """
        self.signal.axes_manager.signal_axes[0].scale = calibration

    def set_diffraction_offset(self, offset):
        """Defines the offset for the signal axis variable s in terms of
        A^-1 per pixel. Note that s is defined here as
        s = 2 sin(theta)/lambda = 1/d, where theta is the scattering angle,
        lambda the wavelength, and d the reciprocal spacing.

        Parameters
        ----------
        offset: float
                    Scattering vector offset in terms of A^-1 per pixel.
        """
        self.signal.axes_manager.signal_axes[0].offset = offset

    def set_s_cutoff(self, s_min, s_max):
        """Scattering vector cutoff for the purposes of fitting an atomic scattering
        factor to the 1D profile. Specified in terms of s (in inverse angstroms).
        s is defined as s = 2 sin(theta)/lambda = 1/d, where theta is the
        scattering angle, lambda the wavelength, and d the reciprocal spacing.

        Parameters
        ----------
        s_min: float
                    Minimum scattering vector amplitude for cutoff.
        s_max: float
                    Maximum scattering vector amplitude for cutoff.
        """
        self.cutoff = [s_min, s_max]
        return

    def fit_atomic_scattering(
        self,
        elements,
        fracs,
        N=1.0,
        C=0.0,
        scattering_factor="lobato",
        plot_fit=True,
        *args,
        **kwargs,
    ):
        """Fits a diffraction intensity profile to the background.

        Uses FIT = N * sum(ci * (fi^2) + C)

        The cutoff for the scattering factor fit to s is defined via the function
        set_s_cutoff above.

        Parameters
        ----------
        elements: list of str
                    A list of elements present (by symbol). No order is necessary.
                    Example: ['Ca', 'C', 'O'] (for CaCO3)
        fracs: list of float
                    A list of fraction of the respective elements. Should sum to 1.
                    Example: [0.2, 0.2, 0.6] (for CaCO3)
        N : float
                    The "slope" of the fit. Initial value is used to start the
                    fit.
        C : float
                    An additive constant to the fit. Initial value is used to
                    start the fit.
        scattering_factor : str
                    Type of scattering parameters fitted. Default is lobato.
                    See scattering_fit_component for more details.
        plot_fit: bool
                    A bool to decide if the fit from scattering is plotted
                    after fitting.
        *args:
            Arguments to be passed to hs.multifit().
        **kwargs:
            Keyword arguments to be passed to hs.multifit().
        """

        fit_model = self.signal.create_model()
        background = scattering_factor_dictionary[scattering_factor](
            elements, fracs, N, C
        )

        fit_model.append(background)
        fit_model.set_signal_range(self.cutoff)
        fit_model.multifit(*args, **kwargs)
        fit_model.reset_signal_range()
        if plot_fit is True:
            fit_model.plot()
        fit = fit_model.as_signal()

        N_values = background.N.as_signal()
        square_sum = background.square_sum

        x_size = N_values.data.shape[0]
        y_size = N_values.data.shape[1]
        normalisation = N_values.data.reshape(x_size, y_size, 1) * square_sum

        self.normalisation = normalisation
        self.background_fit = fit
        return

    def subtract_bkgd_pattern(self, bkgd_pattern, inplace=True, *args, **kwargs):
        """Subtracts a background pattern from the signal.

        This method will edit self.signal.

        Parameters
        ----------
        bkgd_pattern : np.array
            A numpy array of a single line profile of the same resolution
            (same number of pixels) as the radial profile.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """
        return self.signal.map(
            subtract_pattern, pattern=bkgd_pattern, inplace=inplace, *args, **kwargs
        )

    def mask_from_bkgd_pattern(
        self, mask_pattern, mask_threshold=1, inplace=True, *args, **kwargs
    ):
        """Uses a background pattern with a threshold, and sets that part of
        the signal to zero, effectively adding a mask. This can be used to mask
        the central beam.

        Parameters
        ----------
        mask_pattern : np.array
            A numpy array line profile of the same resolution
            as the radial profile.
        mask_threshold : int or float
            An integer or float threshold. Any pixel in the
            mask_pattern with lower intensity is kept, any with
            higher or equal is set to zero.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """

        mask_array = mask_pattern < mask_threshold

        return self.signal.map(
            mask_from_pattern,
            pattern=mask_array.astype(float),
            inplace=inplace,
            *args,
            **kwargs,
        )

    def mask_reduced_intensity(self, mask_pattern, inplace=True, *args, **kwargs):
        """Masks the reduced intensity signal by multiplying it with a pattern
        consisting of only zeroes and ones. This can be used to mask
        the central beam.

        Parameters
        ----------
        mask_pattern : np.array of 0s and 1s
            A numpy array line profile of the same resolution as the radial profile.
            Must consist only of zeroes and ones. Ones are kept while zeroes are
            set to zero.
        mask_threshold : int or float
            An integer or float threshold. Any pixel in the
            mask_pattern with lower intensity is kept, any with
            higher or equal is set to zero.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        """

        mask_array = mask_pattern.astype(np.uint8)
        if np.max(mask_array) != 1 or np.min(mask_array) != 0:
            raise ValueError("Masking array does not consist of zeroes and ones.")

        return self.signal.map(
            mask_from_pattern, pattern=mask_array, inplace=inplace, *args, **kwargs
        )

    def get_reduced_intensity(self):
        """Obtains a reduced intensity profile from the radial profile.

        Parameters
        ----------
        s_cutoff : list of float
                    A list of the form [s_min, s_max] to change the s_cutoff
                    from the fit.

        Returns
        -------
        ri : ReducedIntensity1D

        """

        s = np.arange(self.signal.axes_manager.signal_axes[0].size, dtype="float64")
        s *= self.signal.axes_manager.signal_axes[0].scale

        ri = self.signal._deepcopy_with_new_data(
            (
                2
                * np.pi
                * s
                * np.divide(
                    (self.signal.data - self.background_fit), self.normalisation
                )
            )
        )
        ri.set_signal_type("reduced_intensity")
        title = self.signal.metadata.General.title
        ri.metadata.General.title = f"Reduce intensity of {title}"

        return ri
