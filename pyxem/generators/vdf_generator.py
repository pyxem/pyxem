# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

"""VDF generator and associated tools.

"""

from pyxem.signals.vdf_image import VDFImage
from pyxem.utils.vdf_utils import normalize_vdf

from hyperspy.api import roi
import numpy as np

class VDFGenerator():
    """Generates an VDF images for a specified signal and set of aperture
    positions.

    Parameters
    ----------
    signal : ElectronDiffraction
        The signal of electron diffraction patterns to be indexed.
    vectors: DiffractionVectors
        The vector positions, in calibrated units, at which to position
        integration windows for VDF formation.

    """
    def __init__(self, signal, vectors=None, *args, **kwargs):
        #If ragged the signal axes will not be defined
        self.signal = signal

        if self.vectors = None:
            self.vectors = None
        elif len(vectors.axes_manager.signal_axes)==0:
            unique_vectors = vectors.get_unique_vectors(*args, **kwargs)
            self.vectors = unique_vectors
        else:
            self.vectors = vectors

    def get_vdf_images(self,
                       radius,
                       normalize=False):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        radius : float
            Radius of the integration window in reciprocal angstroms.

        normalize : boolean
            If True each VDF image is normalized so that the maximum intensity
            in each VDF is 1.

        Returns
        -------
        vdfs : Signal2D
            Signal containing virtual dark field images for all unique vectors.
        """
        if self.vectors:
            vdfs = []
            for v in self.vectors.data:
                disk = roi.CircleROI(cx=v[0], cy=v[1], r=radius, r_inner=0)
                vdf = disk(self.signal,
                           axes=self.signal.axes_manager.signal_axes)
                vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)

            vdfim = VDFImage(np.asarray(vdfs))

            if normalize==True:
                vdfim.map(normalize_vdf)

        else:
            raise ValueError("DiffractionVectors non-specified by user. Please"
                             "initialize VDFGenerator with some vectors. "

        return vdfim

    def get_concentric_vdf_images(self,
                                  k_min,
                                  k_max,
                                  k_step,
                                  normalize=False):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        radius : float
            Radius of the integration window in reciprocal angstroms.

        normalize : boolean
            If True each VDF image is normalized so that the maximum intensity
            in each VDF is 1.

        Returns
        -------
        vdfs : Signal2D
            Signal containing virtual dark field images for all unique vectors.
        """
        k0s = np.linspace(k_min, k_max-k_step, k_step)
        k1s = np.linspace(k_min+k_step, k_max, k_step)

        ks = np.array((k0s, k1s)).T

        vdfs = []
        for k in ks:
            annulus = roi.CircleROI(cx=0, cy=0, r=k[1], r_inner=k[0])
            vdf = annulus(self.signal,
                          axes=self.signal.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)

        vdfim = VDFImage(np.asarray(vdfs))

        if normalize==True:
            vdfim.map(normalize_vdf)

        return vdfim
