# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as ss


def _register_drift_5d(data, shifts1, shifts2):
    """
    Register 5D data set using affine transformation

     Parameters
    ----------
    data: np.array or dask.array
        Input image in 5D array (time * ry * rx * kx * ky)
    shifts1: np.array
        1D array for shifts in 1st real space direction or y in hyperspy indexing.
    shifts2: np.array
        1D array for shifts in 2nd real space direction or x in hyperspy indexing.

    Returns
    -------
    data_t: np.array
        5D array after translation according to shift vectors

    """
    data_t = np.zeros_like(data)
    time_size = len(shifts1)
    for i in range(time_size):
        data_t[i, :, :, :, :] = ndi.affine_transform(data[i, :, :, :, :],
                                                     np.identity(4),
                                                     offset=(shifts1[i][0, 0, 0, 0], shifts2[i][0, 0, 0, 0], 0, 0),
                                                     order=1)
    return data_t


def get_drift_vectors(s, **kwargs):
    """
    Calculate real space drift vectors from time series of images

     Parameters
    ----------
    s: Signal2D
        Time series of reconstructed images
    **kwargs:
        Passed to the hs.signals.Signal2D.estimate_shift2D() function

    Returns
    -------
    xshifts: np.array
        1D array of shift in x direction
    yshifts: np.array
        1D array of shift in y direction

    """
    shift_reference = kwargs.get("reference", "stat")
    sub_pixel = kwargs.get("sub_pixel_factor", 10)
    shift_vectors = s.estimate_shift2D(reference=shift_reference,
                                       sub_pixel_factor=sub_pixel,
                                       **kwargs)
    xshifts = shift_vectors[:, 0]
    yshifts = shift_vectors[:, 1]

    return xshifts, yshifts


def _g2_2d(data, normalization='split', kbin=1, tbin=1):
    """
    Calculate k resolved g2(k,t) from I(t,k_r,k_phi)

     Parameters
    ----------
    data: 3D np.array
        Time series for I(t,k_r,k_phi)
    normalization: string
        Normalization format for time autocorrelation, 'split' or 'self'
    kbin: int
        Binning factor for both k axes
    tbin: int
        Binning factor for t axis

    Returns
    -------
    g2: 3D np.array
        Time correlation function g2(t,k_r,k_phi)
    """
    data = data.T
    data = data.reshape((data.shape[0] // kbin), kbin, (data.shape[1] // kbin), kbin, (data.shape[2] // tbin),
                        tbin).sum(5).sum(3).sum(1)

    # Calculate autocorrelation along time axis
    autocorr = ss.fftconvolve(data, data[:, :, ::-1], mode='full', axes=[-1])
    norm = ss.fftconvolve(np.ones(data.shape), data[:, :, ::-1], mode='full', axes=[-1])
    if normalization == 'self':
        overlap_factor = np.expand_dims(np.linspace(data.shape[-1], 1, data.shape[-1]), axis=(0, 1))
        norm_factor = norm[:, :, data.shape[-1]:0:-1] ** 2
        g2 = autocorr[:, :, data.shape[-1] - 1:] / norm_factor * overlap_factor
    if normalization == 'split':
        overlap_factor = np.expand_dims(np.linspace(data.shape[-1], 1, data.shape[-1]), axis=(0, 1))
        norm_factor = norm[:, :, data.shape[-1] - 1:] * norm[:, :, ::-1][:, :, data.shape[-1] - 1:]
        g2 = autocorr[:, :, data.shape[-1] - 1:] / norm_factor * overlap_factor
    else:
        raise ValueError(
            normalization + " not recognize, normalization must be chosen 'split' or 'self'"
        )

    return g2.T
