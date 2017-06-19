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

from __future__ import division

from hyperspy.api import roi
from hyperspy.signals import BaseSignal

from pycrystem.utils.expt_utils import *

"""
Signal class for diffraction vectors.
"""


class DiffractionVectors(BaseSignal):
    _signal_type = "diffraction_vectors"

    def __init__(self,
                 calibration,
                 *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        self.calibration = calibration

    def get_gvector_magnitudes(self, center):
        """Calculate the magnitude of diffraction vectors.

        Parameters
        ----------

        center :

        Returns
        -------

        gmagnitudes : array
            Array

        """
        # Allocate an empty array in which to store the gvector magnitudes.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gvectors = np.zeros(arr_shape, dtype=object)
        #
        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            centered = peaks[it] - center
            for j in np.arange(len(centered)):
                res.append(np.linalg.norm(centered[j]))

            cent = peaks[it][np.where(res == min(res))[0]][0]
            vectors = peaks[it] - cent

            mags = []
            for k in np.arange(len(vectors)):
                mags.append(np.linalg.norm(vectors[k]))
            maga = np.asarray(mags)
            gvectors[it] = maga * self.axes_manager.signal_axes[0].scale
        return gvectors

    def get_glength_histogram(self, bins):
        """Obtain a histogram of all measured diffraction vector magnitudes.
        """
        gmags = self.get_gvector_magnitudes()

        glist = []
        for i in dp.axes_manager:
            it = (i[1], i[0])
            g = gmags[it]
            for j in np.arange(len(g)):
                glist.append(g[j])
        gs = np.asarray(glist)
        gsig = hs.signals.Signal1D(gs)
        return gsig.get_histogram(bins=bins)

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


        return gindex

    def get_zone_axis_indexation(self):
        """Determine the zone axis consistent with the majority of indexed
        diffraction vectors.

        Parameters
        ----------

        Returns
        -------

        """

    def get_unique_vectors(self):
        """Obtain a unique list of diffraction vectors.

        Returns
        -------
        unique_vectors : list
            Unique list of all diffraction vectors.
        """
        #Create empty list
        gv = []
        #Iterate through vectors
        for i in dp.axes_manager:
            it = (i[1], i[0])
            g = peaks[it]
            for j in np.arange(len(g)):
                #if vector in list pass else add list
                if np.asarray(g[j]) in np.asarray(gv):
                    pass
                else:
                    gv.append(g[j])
        return gv

    def get_reflection_intensities(self,
                                   unique_vectors=None,
                                   electron_diffraction,
                                   radius):
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
        """
        if unique_vectors==None:
            unique_vectors = self.get_unique_vectors()

        cs = np.asarray(unique_vectors)
        cs = cs * electron_diffraction.axes_manager.signal_axes[0].scale
        cs = cs + electron_diffraction.axes_manager.signal_axes[0].offset

        vdfs = []
        for i in np.arange(len(gvuna)):
            roi = hs.roi.CircleROI(cx=cs[i][1], cy=cs[i][0],
                                   r=radius, r_inner=0)
            vdf = roi(electron_diffraction, axes=electron_diffraction.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)
        return hs.signals.Signal2D(np.asarray(vdfs))
