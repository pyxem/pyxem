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
    def __init__(self, signal, vectors, *args, **kwargs):
        #If ragged the signal axes will not be defined
        if len(vectors.axes_manager.signal_axes)==0:
            unique_vectors = vectors.get_unique_vectors(*args, **kwargs)

        else:
            unique_vectors = vectors

        self.signal = signal
        self.vectors = unique_vectors

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
        vdfs = []
        for v in self.vectors.data:
            disk = roi.CircleROI(cx=v[1], cy=v[0], r=radius, r_inner=0)
            vdf = disk(self.signal,
                       axes=self.signal.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)

        vdfim = VDFImage(np.asarray(vdfs))

        if normalize==True:
            vdfim.map(normalize_vdf)

        return vdfim
