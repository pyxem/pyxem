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
"""Signal class for virtual diffraction contrast images.

"""

from pyxem.signals import push_metadata_through

import numpy as np
from tqdm import tqdm
from hyperspy.signals import BaseSignal, Signal2D
from pyxem.utils.vdf_utils import (normalize_vdf, norm_cross_corr, get_vector_i)
from pyxem.signals.diffraction_vectors import DiffractionVectors

class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vectors = None

class VDFSegment:
    _signal_type = "vdf_segment"

    def __init__(self, segments, vectors_of_segments,
                 intensities_of_segments=None, *args,**kwargs):
        # Segments as Signal2D
        self.segments = segments
        # DiffractionVectors
        self.vectors_of_segments = vectors_of_segments

    def correlate_segments(self, corr_threshold=0.9):
        """Iterates through VDF segments and sums those two segments
        that have the higest normalised cross correlation, if it is
        above the threshold corr_threshold. The vectors of each segment
        sum will be updated accordingly, so that the vectors of each
        resulting segment sum are all the vectors of the original
        individual segments.

        Parameters
        ----------
        corr_threshold: float
            Segments will be summed if they have a normalized cross-
            correlation above corr_threshold. Must be between 0 and 1.

        Returns
        -------
        VDFSegment
            The VDFSegment instance updated according to the image
            correlation results.
        """
        image_stack = self.segments.data.copy()
        vectors = self.vectors_of_segments.data

        if len(np.shape(vectors)) <= 1:
            raise ValueError("Input vectors are not of correct shape. Try to "
                             "rerun correlate_segments on original VDFSegment "
                             "resulting from get_vdf_segments.")

        num_vectors = np.shape(vectors)[0]
        gvectors = np.array(np.empty(num_vectors, dtype=object))
        for i in np.arange(num_vectors):
            gvectors[i] = np.array(vectors[i].copy())

        i = 0
        pbar = tqdm(total=np.shape(image_stack)[0])
        while np.shape(image_stack)[0] > i:
            # For each segment, calculate the normalized cross-correlation to
            # all other segments, and define add_index referring to that with
            # the largest correlation.
            corr_list = list(map(
                lambda x: norm_cross_corr(x, template=image_stack[i]),
                image_stack))
            corr_max = np.max(corr_list)
            add_index = np.where(corr_list == corr_max)

            # If the largest correlation value is above the threshold, add
            # current segment to the segment with the largest correlation.
            if corr_max > corr_threshold:
                image_stack[i] = np.sum(image_stack[i], image_stack[add_index],
                                        axis=0)
                gvectors[i] = get_vector_i(gvectors[add_index], add_index,
                                           gvectors[i])
                # Delete the segment and vectors that were added to those at i.
                image_stack = np.delete(image_stack, add_index, axis=0)
                gvectors = np.delete(gvectors, add_index, axis=0)

            if add_index > i:
                i = i + 1

            pbar.update(1)

        pbar.close()

        return VDFSegment(Signal2D(image_stack), DiffractionVectors(gvectors).T)

    def get_virtual_electron_diffraction_signal(self,
                                                electron_diffraction,
                                                distance_threshold=None,
                                                A = 255):
        """ Created an ElectronDiffraction signal consisting of Gaussians at all the gvectors.
        Parameters
        ---------- 
        electron_diffraction: ElectronDiffraction
            ElectronDiffraction signal that the merge_stack_corr originates from.  
        distance_threshold : float
            The FWHM in the 2D Gaussians that will be calculated for each g-vector,
            in order to create the virtual DPs. It is reasonable to choose this value equal to
            the distance_threshold that was used to find the unique g-vectors, and thus the name. 
            
        Returns
        -------
        gvec_sig: ElectronDiffraction
            ElectronDiffraction signal based on the gvectors.
        """
        from pycrystem.diffraction_signal import ElectronDiffraction

        gvector_stack=self.vectors.data
        num_of_im=np.shape(gvector_stack)[0]
        
        size_x = electron_diffraction.axes_manager[2].size
        size_y = electron_diffraction.axes_manager[3].size
        cx = electron_diffraction.axes_manager[2].offset
        cy = electron_diffraction.axes_manager[3].offset
        scale_x = electron_diffraction.axes_manager[2].scale
        scale_y = electron_diffraction.axes_manager[3].scale

        DP_sig = np.zeros((size_x, size_y, num_of_im))
        X,Y=np.indices((size_x,size_y))
        X=X*scale_x + cx
        Y=Y*scale_y + cy
        
        if distance_threshold == None:
            distance_threshold = np.max((scale_x,scale_y))
        
        for i in range(num_of_im):
            if len(np.shape(gvector_stack[i]))>1:
                for n in gvector_stack[i]:
                    DP_sig[...,i] = DP_sig[...,i] + A * np.exp(-4*np.log(2) * ((X-n[1])**2 +(Y-n[0])**2)/distance_threshold**2)
            else: 
                DP_sig[...,i] = DP_sig[...,i] + A * np.exp(-4*np.log(2) * ((X-gvector_stack[i][1])**2 +(Y-gvector_stack[i][0])**2)/distance_threshold**2)
        gvec_sig = ElectronDiffraction(DP_sig.T)
        gvec_sig.axes_manager[1].scale=electron_diffraction.axes_manager[2].scale
        gvec_sig.axes_manager[1].units=electron_diffraction.axes_manager[2].units
        gvec_sig.axes_manager[2].scale=electron_diffraction.axes_manager[2].scale
        gvec_sig.axes_manager[2].units=electron_diffraction.axes_manager[2].units
            
        return gvec_sig

    def threshold_segments(self, image_number_threshold=None,
                           vector_number_threshold=None):

        if image_number_threshold is None and vector_number_threshold is None:
            raise ValueError("Specify image number and/or vector number \
             threshold.")

        image_stack = self.segments.data.copy()
        vectors = self.vectors_of_segments.data.copy()

        if image_number_threshold is not None:
            n = 0
            while np.shape(image_stack)[0] > n:
                if np.max(image_stack[n]) < image_number_threshold:
                    image_stack = np.delete(image_stack, n, axis=0)
                    vectors = np.delete(vectors, n, axis=0)
                else:
                    n=n+1

        if vector_number_threshold is not None:
            n = 0
            while np.shape(image_stack)[0] > n:
                if np.shape(vectors[n])[0] < vector_number_threshold:
                    image_stack = np.delete(image_stack, n, axis=0)
                    vectors = np.delete(vectors, n, axis=0)
                else:
                    n = n+1

        if not np.any(image_stack):
            print('No stack left after thresholding. Check thresholds.')
            return 0

        return VDFSegment(Signal2D(image_stack), DiffractionVectors(vectors))
