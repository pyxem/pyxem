# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

from hyperspy.api import roi
from hyperspy.signals import BaseSignal, Signal1D, Signal2D

from .utils.expt_utils import *

"""
Signal class for diffraction vectors.
"""

def _calculate_norms(z):
    norms = []
    #print(z)
    for i in z[0]:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)

class DiffractionVectors(BaseSignal):
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
""" # This overwrites a method in base signal and should be renamed
    def plot(self):
        """Plot the diffraction vectors.
        """
        #Find the unique gvectors to plot.
        unique_vectors = self.get_unique_vectors()
        #Plot the gvector positions
        import matplotlib.pyplot as plt
        plt.plot(unique_vectors.T[1], unique_vectors.T[0], 'ro')
        plt.axes().set_aspect('equal')
        plt.show()
"""
    def get_magnitudes(self):
        """Calculate the magnitude of diffraction vectors.

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containging an array of gvector magnitudes at each
            navigation position.

        """
        magnitudes = self.map(_calculate_norms, inplace=False)
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
        gnorms = self.get_magnitudes()

        glist=[]
        for i in gnorms._iterate_signal():
            for j in np.arange(len(i[0])):
                glist.append(i[0][j])
        gs = np.asarray(glist)
        gsig = Signal1D(gs)
        ghis = gsig.get_histogram(bins=bins)
        ghis.axes_manager.signal_axes[0].name = 'g-vector magnitude'
        ghis.axes_manager.signal_axes[0].units = '$A^{-1}$'
        return ghis

    def get_unique_vectors(self,
                           distance_threshold=0):
        """Obtain a list of unique diffraction vectors.

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to be considered
            as unique diffraction vectors.

        Returns
        -------
        unique_vectors : float
            Ndarray of all unique diffraction vectors.
        """
        from scipy.spatial import distance_matrix
        def get_new_indices_from_distance_matrix(distances,distance_threshold):
        #Checks if the distances from one vector in vlist to all other vectors in gvlist
        #is larger than distance_threshold. 
            new_indices = []
            l = np.shape(distances)[0]
            for i in range(np.shape(distances)[1]):
                if (np.sum(distances[:,i] > distance_threshold) == l):
                    new_indices = np.append(new_indices, i)
            return np.array(new_indices,dtype=np.int)
        
        if (self.axes_manager.navigation_dimension == 2):
            gvlist = np.array([self.data[0,0][0]])
        else:
            gvlist = np.array([self.data[0][0]])
            
        for i in tqdm(self._iterate_signal()):
            vlist = i[0]
            distances = distance_matrix(gvlist,vlist)
            new_indices = get_new_indices_from_distance_matrix(distances,distance_threshold)
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
        return np.delete(gvlist,delete_indices,axis = 0)

    def get_vdf_images(self,
                       electron_diffraction,
                       radius,
                       unique_vectors=None):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        unique_vectors : list (optional)
            Unique list of diffracting vectors if pre-calculated. If None the
            unique vectors in self are determined and used.

        electron_diffraction : ElectronDiffraction
            ElectronDiffraction signal from which to extract the reflection
            intensities.

        radius : float
            Radius of the integration window summed over in reciprocal angstroms.

        Returns
        -------
        vdfs : Signal2D
            Signal containing virtual dark field images for all unique g-vectors.
        """
        if unique_vectors==None:
            unique_vectors = self.get_unique_vectors()
        else:
            unique_vectors = unique_vectors

        vdfs = []
        for v in unique_vectors:
            disk = roi.CircleROI(cx=v[1], cy=v[0], r=radius, r_inner=0)
            vdf = disk(electron_diffraction,
                       axes=electron_diffraction.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)
        return Signal2D(np.asarray(vdfs))

    def get_gvector_indexation(self,
                               calculated_peaks,
                               magnitude_threshold,
                               angular_threshold=None):
        """Index diffraction vectors based on the magnitude of individual
        vectors and optionally the angles between pairs of vectors.

        Parameters
        ----------

        calculated_peaks : array
            Structured array containing the theoretical diffraction vector
            magnitudes and angles between vectors.

        magnitude_threshold : Float
            Maximum deviation in diffraction vector magnitude from the
            theoretical value for an indexation to be considered possible.

        angular_threshold : float
            Maximum deviation in the measured angle between vector
        Returns
        -------

        gindex : array
            Structured array containing possible indexations
            consistent with the data.

        """
        #TODO: Specify threshold as a fraction of the g-vector magnitude.
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

    def get_zone_axis_indexation(self):
        """Determine the zone axis consistent with the majority of indexed
        diffraction vectors.

        Parameters
        ----------

        Returns
        -------

        """
