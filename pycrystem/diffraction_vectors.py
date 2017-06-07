# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
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

import tqdm
from hyperspy.api import interactive
from hyperspy.api import roi
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal2D, Signal1D, BaseSignal

from pycrystem.utils.expt_utils import *
from .library_generator import DiffractionLibrary
from .indexation_generator import IndexationGenerator

"""
Signal class for diffraction vectors
"""


class DiffractionVectors(BaseSignal):
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        # Attributes defaults

    def get_gvector_magnitudes(self, peaks):
        """Obtain the magnitude of g-vectors in calibrated units
        from a structured array containing peaks in array units.

        Parameters
        ----------

        Returns
        -------

        """
        # Allocate an empty structured array in which to store the gvector
        # magnitudes.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gvectors = np.zeros(arr_shape, dtype=object)
        #
        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            centered = peaks[it] - [256,256]
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

    def get_reflection_intensities(self, indexed_reflections):
        """

        Parameters
        ----------

        Returns
        -------
        """
        #TODO: Implement sum in ROI around peak centers.
        pass

    def get_gvector_indexation(self, glengths, calc_peaks, threshold):
        """Index the magnitude of g-vectors in calibrated units
        from a structured array containing gvector magnitudes.

        Parameters
        ----------

        glengths : A structured array containing the

        calc_peaks : A structured array

        threshold : Float indicating the maximum allowed deviation from the
            theoretical value.

        Returns
        -------

        gindex : Structured array containing possible indexation results
            consistent with the data.

        """
        # TODO: Make it so that the threshold can be specified as a fraction of
        # the g-vector magnitude.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gindex = np.zeros(arr_shape, dtype=object)

        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            for j in np.arange(len(glengths[it])):
                peak_diff = (calc_peaks.T[1] - glengths[it][j]) * (calc_peaks.T[1] - glengths[it][j])
                res.append((calc_peaks[np.where(peak_diff < threshold)],
                            peak_diff[np.where(peak_diff < threshold)]))
            gindex[it] = res

        return gindex
