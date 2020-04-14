# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

"""Variance generators in real and reciprocal space for fluctuation electron
microscopy.

"""

import numpy as np
from hyperspy.signals import Signal2D
from hyperspy.api import stack

from pyxem.signals.diffraction_variance2d import DiffractionVariance2D
from pyxem.signals.diffraction_variance2d import ImageVariance
from pyxem.signals import transfer_signal_axes
from pyxem.signals import transfer_navigation_axes_to_signal_axes

import matplotlib.pylab as plt
import numpy as np
from dask import delayed
from tqdm import tqdm
import hyperspy.api as hs


def fem_calc(s, centre_x=None, centre_y=None, show_progressbar=True):
    """Perform analysis of fluctuation electron microscopy (FEM) data
    as outlined in:

    T. L. Daulton, et al., Ultramicroscopy 110 (2010) 1279-1289.
    doi:10.1016/j.ultramic.2010.05.010

    Parameters
    ----------
    s : PixelatedSTEM
        Signal on which FEM analysis was performed
    centre_x, centre_y : int, optional
        All the diffraction patterns assumed to have the same
        centre position.

    show_progressbar : bool
        Default True

    Returns
    -------
    results : Python dictionary
        Results of FEM data analysis, including the normalized variance
        of the annular mean (V-Omegak), mean of normalized variances of
        rings (V-rk), normalized variance of ring ensemble (Vrek),
        the normalized variance image (Omega-Vi), and annular mean of
        the variance image (Omega-Vk).

    Examples
    --------
    >>> import pixstem.dummy_data as dd
    >>> import pixstem.fem_tools as femt
    >>> s = dd.get_fem_signal()
    >>> fem_results = femt.fem_calc(
    ...     s,
    ...     centre_x=128,
    ...     centre_y=128,
    ...     show_progressbar=False)
    >>> fem_results['V-Omegak'].plot()

    """
    offset = False

    if centre_x is None:
        centre_x = np.int(s.axes_manager.signal_shape[0] / 2)

    if centre_y is None:
        centre_y = np.int(s.axes_manager.signal_shape[1] / 2)

    if s.data.min() == 0:
        s.data += 1  # To avoid division by 0
        offset = True
    results = dict()

    results["RadialInt"] = s.radial_average(
        centre_x=centre_x,
        centre_y=centre_y,
        normalize=False,
        show_progressbar=show_progressbar,
    )

    radialavgs = s.radial_average(
        centre_x=centre_x,
        centre_y=centre_y,
        normalize=True,
        show_progressbar=show_progressbar,
    )
    if radialavgs.data.min() == 0:
        radialavgs.data += 1

    results["V-Omegak"] = ((radialavgs ** 2).mean() / (radialavgs.mean()) ** 2) - 1
    results["RadialAvg"] = radialavgs.mean()

    if s._lazy:
        results["Omega-Vi"] = ((s ** 2).mean() / (s.mean()) ** 2) - 1
        results["Omega-Vi"].compute(progressbar=show_progressbar)
        results["Omega-Vi"] = pixstem.pixelated_stem_class.PixelatedSTEM(
            results["Omega-Vi"]
        )

        results["Omega-Vk"] = results["Omega-Vi"].radial_average(
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=True,
            show_progressbar=show_progressbar,
        )

        oldshape = None
        if len(s.data.shape) == 4:
            oldshape = s.data.shape
            s.data = s.data.reshape(
                s.data.shape[0] * s.data.shape[1], s.data.shape[2], s.data.shape[3]
            )
        y, x = np.indices(s.data.shape[-2:])
        r = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
        r = r.astype(np.int)

        nr = np.bincount(r.ravel())
        Vrklist = []
        Vreklist = []

        for k in tqdm(range(0, len(nr)), disable=(not show_progressbar)):
            locs = np.where(r == k)
            vals = s.data.vindex[:, locs[0], locs[1]].T
            Vrklist.append(np.mean((np.mean(vals ** 2, 1) / np.mean(vals, 1) ** 2) - 1))
            Vreklist.append(np.mean(vals.ravel() ** 2) / np.mean(vals.ravel()) ** 2 - 1)

        Vrkdask = delayed(Vrklist)
        Vrekdask = delayed(Vreklist)

        results["Vrk"] = hs.signals.Signal1D(
            Vrkdask.compute(progressbar=show_progressbar)
        )
        results["Vrek"] = hs.signals.Signal1D(
            Vrekdask.compute(progressbar=show_progressbar)
        )
    else:
        results["Omega-Vi"] = ((s ** 2).mean() / (s.mean()) ** 2) - 1
        results["Omega-Vk"] = results["Omega-Vi"].radial_average(
            centre_x=centre_x,
            centre_y=centre_y,
            normalize=True,
            show_progressbar=show_progressbar,
        )
        oldshape = None
        if len(s.data.shape) == 4:
            oldshape = s.data.shape
            s.data = s.data.reshape(
                s.data.shape[0] * s.data.shape[1], s.data.shape[2], s.data.shape[3]
            )
        y, x = np.indices(s.data.shape[-2:])
        r = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
        r = r.astype(np.int)

        nr = np.bincount(r.ravel())
        results["Vrk"] = np.zeros(len(nr))
        results["Vrek"] = np.zeros(len(nr))

        for k in tqdm(range(0, len(nr)), disable=(not show_progressbar)):
            locs = np.where(r == k)
            vals = s.data[:, locs[0], locs[1]]
            results["Vrk"][k] = np.mean(
                (np.mean(vals ** 2, 1) / np.mean(vals, 1) ** 2) - 1
            )
            results["Vrek"][k] = (
                np.mean(vals.ravel() ** 2) / np.mean(vals.ravel()) ** 2 - 1
            )

        results["Vrk"] = hs.signals.Signal1D(results["Vrk"])
        results["Vrek"] = hs.signals.Signal1D(results["Vrek"])

    if oldshape:
        s.data = s.data.reshape(oldshape)
    if offset:
        s.data -= 1  # Undo previous addition of 1 to input data
    return results


class VarianceGenerator:
    """Generates variance images for a specified signal and set of aperture
    positions.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.

    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal

        # add a check for calibration

    def get_diffraction_variance(self, dqe, set_data_type=None):
        """Calculates the variance in scattered intensity as a function of
        scattering vector.

        Parameters
        ----------
        dqe : float
            Detective quantum efficiency of the detector for Poisson noise
            correction.
        data_type : numpy data type.
            For numpy data types, see
            https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html.
            This is incorporated as squaring the numbers in meansq_dp results
            in considerably larger than the ones in the original array. This can
            result in an overflow error that is difficult to distinguish. Hence
            the data can be converted to a different data type to accommodate.

        Returns
        -------
        vardps : DiffractionVariance2D
            A DiffractionVariance2D object containing the mean DP, mean
            squared DP, and variance DP.
        """

        dp = self.signal
        mean_dp = dp.mean((0, 1))
        if set_data_type is None:
            meansq_dp = Signal2D(np.square(dp.data)).mean((0, 1))
        else:
            meansq_dp = Signal2D(np.square(dp.data.astype(set_data_type))).mean((0, 1))

        normvar = (meansq_dp.data / np.square(mean_dp.data)) - 1.0
        var_dp = Signal2D(normvar)
        corr_var_array = var_dp.data - (np.divide(dqe, mean_dp.data))
        corr_var_array[np.isinf(corr_var_array)] = 0
        corr_var_array[np.isnan(corr_var_array)] = 0
        corr_var = Signal2D(corr_var_array)
        vardps = stack((mean_dp, meansq_dp, var_dp, corr_var))
        sig_x = vardps.data.shape[1]
        sig_y = vardps.data.shape[2]

        dv = DiffractionVariance2D(vardps.data.reshape((2, 2, sig_x, sig_y)))

        dv = transfer_signal_axes(dv, self.signal)

        return dv

    def get_image_variance(self, dqe):
        """Calculates the variance in scattered intensity as a function of
        scattering vector. The calculated variance is normalised by the mean
        squared, as is appropriate for the distribution of intensities. This
        causes a problem if Poisson noise is significant in the data, resulting
        in a divergence of the Poisson noise term. To in turn remove this
        effect, we subtract a dqe/mean_dp term (although it is suggested that
        dqe=1) from the data, creating a "poisson noise-free" corrected variance
        pattern. DQE is fitted to make this pattern flat.

        Parameters
        ----------

        dqe : float
            Detective quantum efficiency of the detector for Poisson noise
            correction.

        Returns
        -------

        varims : ImageVariance
            A two dimensional Signal class object containing the mean DP, mean
            squared DP, and variance DP, and a Poisson noise-corrected variance
            DP.
        """
        im = self.signal.T
        mean_im = im.mean((0, 1))
        meansq_im = Signal2D(np.square(im.data)).mean((0, 1))
        normvar = (meansq_im.data / np.square(mean_im.data)) - 1.0
        var_im = Signal2D(normvar)
        corr_var_array = normvar - (np.divide(dqe, mean_im.data))
        corr_var_array[np.invert(np.isfinite(corr_var_array))] = 0
        corr_var = Signal2D(corr_var_array)
        varims = stack((mean_im, meansq_im, var_im, corr_var))

        sig_x = varims.data.shape[1]
        sig_y = varims.data.shape[2]
        iv = ImageVariance(varims.data.reshape((2, 2, sig_x, sig_y)))
        iv = transfer_navigation_axes_to_signal_axes(iv, self.signal)

        return iv
