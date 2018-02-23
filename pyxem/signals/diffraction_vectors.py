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

from hyperspy.api import roi
from hyperspy.signals import BaseSignal, Signal1D, Signal2D

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from pyxem.signals.vdf_image import VDFImage

from pyxem.utils.expt_utils import *
from pyxem.utils.vector_utils import *
from pyxem.utils.vdf_utils import normalize_vdf

"""
Signal class for diffraction vectors.

There are two cases that are supported:

1. A map of diffraction vectors, which will in general be a ragged signal of
signals. It the navigation dimensions of the map and contains a signal for each
peak at every position.

2. A list of diffraction vectors with dimensions < n | 2 > where n is the
number of peaks.
"""


class DiffractionVectors(BaseSignal):
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)

    def plot_diffraction_vectors(self, xlim, ylim):
        """Plot the unique diffraction vectors.
        """
        #Find the unique gvectors to plot.
        unique_vectors = self.get_unique_vectors()
        #Plot the gvector positions
        plt.plot(unique_vectors.data.T[1], unique_vectors.data.T[0], 'ro')
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.axes().set_aspect('equal')
        plt.show()

    def get_magnitudes(self, *args, **kwargs):
        """Calculate the magnitude of diffraction vectors.

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containging an array of gvector magnitudes at each
            navigation position.

        """
        #If ragged the signal axes will not be defined
        if len(self.axes_manager.signal_axes)==0:
            magnitudes = self.map(calculate_norms_ragged,
                                  inplace=False,
                                  *args, **kwargs)
        #Otherwise easier to calculate.
        else:
            magnitudes = BaseSignal(calculate_norms(self))
            magnitudes.axes_manager.set_signal_dimension(0)

        return magnitudes

    def get_magnitude_histogram(self, bins):
        """Obtain a histogram of gvector magnitudes.

        Parameters
        ----------
        bins : numpy array
            The bins to be used to generate the histogram.

        Returns
        -------
        ghist : Signal1D
            Histogram of gvector magnitudes.

        """
        gmags = self.get_magnitudes()

        if len(self.axes_manager.signal_axes)==0:
            glist=[]
            for i in gmags._iterate_signal():
                for j in np.arange(len(i[0])):
                    glist.append(i[0][j])
            gs = np.asarray(glist)
            gsig = Signal1D(gs)
            ghis = gsig.get_histogram(bins=bins)

        else:
            ghis = gmags.get_histogram(bins=bins)

        ghis.axes_manager.signal_axes[0].name = 'g-vector magnitude'
        ghis.axes_manager.signal_axes[0].units = '$A^{-1}$'

        return ghis

    def get_unique_vectors(self,
                           distance_threshold=0):
        """Obtain the unique diffraction vectors.

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to be
            considered unique diffraction vectors.

        Returns
        -------
        unique_vectors : DiffractionVectors
            A DiffractionVectors object containing only the unique diffraction
            vectors in the original object.
        """
        if (self.axes_manager.navigation_dimension == 2):
            gvlist = np.array([self.data[0,0][0]])
        else:
            gvlist = np.array([self.data[0][0]])

        for i in self._iterate_signal():
            vlist = i[0]
            distances = distance_matrix(gvlist, vlist)
            new_indices = get_indices_from_distance_matrix(distances,
                                                           distance_threshold)
            gvlist_new = vlist[new_indices]
            if gvlist_new.any():
                gvlist=np.concatenate((gvlist, gvlist_new),axis=0)

        #An internal check, just to be sure.
        delete_indices = []
        l = np.shape(gvlist)[0]
        distances = distance_matrix(gvlist,gvlist)
        for i in range(np.shape(distances)[1]):
            if (np.sum(distances[:,i] <= distance_threshold) > 1):
                delete_indices = np.append(delete_indices, i)
        gvecs = np.delete(gvlist, delete_indices,axis = 0)

        #Manipulate into DiffractionVectors class
        unique_vectors = DiffractionVectors(gvecs)
        unique_vectors.axes_manager.set_signal_dimension(1)

        return unique_vectors


    def get_vdf_images(self,
                       electron_diffraction,
                       radius,
                       normalize=False):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        electron_diffraction : ElectronDiffraction
            ElectronDiffraction signal from which to extract the reflection
            intensities.

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
        #If ragged the signal axes will not be defined
        if len(self.axes_manager.signal_axes)==0:
            unique_vectors = self.get_unique_vectors()

        else:
            unique_vectors = self

        vdfs = []
        for v in unique_vectors.data:
            disk = roi.CircleROI(cx=v[1], cy=v[0], r=radius, r_inner=0)
            vdf = disk(electron_diffraction,
                       axes=electron_diffraction.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)

        vdfim = VDFImage(np.asarray(vdfs))

        if normalize==True:
            vdfim.map(normalize_vdf)

        return vdfim

    def get_diffracting_pixels_map(self, binary=False):
        """Map of the number of vectors at each navigation position.

        Parameters
        ----------
        binary : boolean
            If True a binary image with diffracting pixels taking value == 1 is
            returned.

        Returns
        -------
        crystim : Signal2D
            2D map of diffracting pixels.
        """
        crystim = self.map(get_npeaks, inplace=False).as_signal2D((0,1))

        if binary==True:
            crystim = crystim == 1

        crystim.change_dtype('float')

        return crystim

    def get_gvector_indexation(self,
                               structure,
                               magnitude_threshold,
                               angular_threshold=None,
                               maximum_length=1):
        """Index diffraction vectors based on the magnitude of individual
        vectors and optionally the angles between pairs of vectors.

        Parameters
        ----------
        structure : Structure
            pymatgen structure to be used for indexation

        magnitude_threshold : Float
            Maximum deviation in diffraction vector magnitude from the
            theoretical value for an indexation to be considered possible.

        angular_threshold : float
            Maximum deviation in the measured angle between vector

        maximum_length : float
            Maximum g-vector length to included in indexation.

        Returns
        -------
        gindex : array
            Structured array containing possible indexations
            consistent with the data.

        """
        #TODO: Specify threshold as a fraction of the g-vector magnitude.
        recip_latt = structure.lattice.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], maximum_length)
        calc_peaks = np.asarray(sorted(recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])))

        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gindex = np.zeros(arr_shape, dtype=object)

        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            for j in np.arange(len(glengths[it])):
                peak_diff = (calc_peaks.T[1] - glengths[it][j]) * (calc_peaks.T[1] - glengths[it][j])
                res.append((calc_peaks[np.where(peak_diff < magnitude_threshold)],
                            peak_diff[np.where(peak_diff < magnitude_threshold)]))
            gindex[it] = res

        if angular_threshold==None:
            pass
        else:
            pass

        return gindex
