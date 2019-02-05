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
from hyperspy.signals import Signal2D
from pyxem.utils.vdf_utils import (norm_cross_corr, corr_check,
                                   get_vectors_and_indices_i, get_gaussian2d)
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction import ElectronDiffraction


class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vectors = None

class VDFSegment:
    _signal_type = "vdf_segment"

    def __init__(self, segments, vectors_of_segments, intensities=None,
                 *args,**kwargs):
        # Segments as Signal2D
        self.segments = segments
        # DiffractionVectors
        self.vectors_of_segments = vectors_of_segments
        # Intensities corresponding to each vector
        self.intensities = intensities

    def correlate_segments(self, corr_threshold=0.9):
        """Iterates through VDF segments and sums those that are
        associated with the same segment. Summation will be done for
        those segments that have a normalised cross correlation above
        corr_threshold. The vectors of each segment sum will be updated
        accordingly, so that the vectors of each resulting segment sum
        are all the vectors of the original individual segments. Each
        vector is assigned an intensity that is the integrated intensity
        of the segment it originates from.

        Parameters
        ----------
        corr_threshold : float
            Segments will be summed if they have a normalized cross-
            correlation above corr_threshold. Must be between 0 and 1.

        Returns
        -------
        VDFSegment
            The VDFSegment instance updated according to the image
            correlation results.
        """

        vectors = self.vectors_of_segments.data
        if len(np.shape(vectors)) <= 1:
            raise ValueError("Input vectors are not of correct shape. Try to "
                             "rerun correlate_segments on original VDFSegment "
                             "resulting from get_vdf_segments.")

        segments = self.segments.data
        image_stack = self.segments.data.copy()
        num_vectors = np.shape(vectors)[0]
        gvectors = np.array(np.empty(num_vectors, dtype=object))
        vector_indices = np.array(np.empty(num_vectors, dtype=object))

        for i in np.arange(num_vectors):
            gvectors[i] = np.array(vectors[i].copy())
            vector_indices[i] = np.array([i], dtype=int)

        i = 0
        pbar = tqdm(total=np.shape(image_stack)[0])
        while np.shape(image_stack)[0] > i:
            # For each segment, calculate the normalized cross-correlation to
            # all other segments, and define add_indices for those with a value
            # above corr_threshold.
            corr_list = list(map(
                lambda x: norm_cross_corr(x, template=image_stack[i]),
                image_stack))
            corr_add = list(map(
                lambda x: corr_check(x, corr_threshold=corr_threshold),
                corr_list))
            add_indices = np.where(corr_add)

            # If there are more add_indices than 1 (i.e. more segments should be
            # added than itself), sum segments and add their vectors.
            if np.shape(add_indices[0])[0] > 1:
                image_stack[i] = np.sum(
                    list(map(lambda x: np.sum([x, image_stack[i]], axis=0),
                             image_stack[add_indices])), axis=0)
                add_indices = add_indices[0]
                gvectors[i], vector_indices[i] = get_vectors_and_indices_i(
                    gvectors[add_indices], gvectors[i],
                    vector_indices[add_indices],
                    vector_indices[i])

                # Delete the segment and vectors that were added to other
                # segments and vectors, except the current one at i.
                add_indices_noi = np.delete(add_indices,
                                            np.where(add_indices == i),
                                            axis=0)
                image_stack = np.delete(image_stack, add_indices_noi, axis=0)
                gvectors = np.delete(gvectors, add_indices_noi, axis=0)
                vector_indices = np.delete(vector_indices, add_indices_noi,
                                           axis=0)

            else:
                add_indices_noi = add_indices

            # Update the iterator value i, after deleting elements.
            if np.where(add_indices == i) != np.array([0]):
                i = i + 1 - (np.shape(np.where(add_indices < i))[1])
            else:
                i = i + 1
            pbar.update(np.shape(add_indices_noi)[0])

        pbar.close()

        # Sum the intensities in the original segments and assign those to the
        # correct vectors by referring to vector_indices.
        segment_intensities = np.sum(segments, axis=(1, 2))
        gvector_intensities = np.array(np.empty(len(gvectors)), dtype=object)
        for i in range(len(gvectors)):
            gvector_intensities[i] = segment_intensities[vector_indices[i]]

        return VDFSegment(Signal2D(image_stack), DiffractionVectors(gvectors).T,
                          gvector_intensities)

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

    def get_virtual_electron_diffraction(self, calibration, shape, sigma=None):
        """ Obtain a virtual electron diffraction signal that consists
        of one virtual diffraction pattern for each segment. The virtual
        diffraction pattern is composed of Gaussians centered at each
        vector position. If given, the integrated intensities of each
        vector will be taken into account by multiplication with the
        Gaussians.

        Parameters
        ----------
        calibration : float
            Reciprocal space calibration in inverse Angstrom per pixel.
        shape : tuple
            Shape of the signal, (shape_x, shape_y) in pixels, where
            shape_x and shape_y are integers.
        sigma : float
            The standard deviation of the Gaussians in inverse Angstrom
            per pixel.

        Returns
        -------
        virtual_ed : ElectronDiffraction
            Virtual electron diffraction signal consisting of one
            virtual diffraction pattern for each segment.
        """
        # TODO : Update all axes, scales, offsets etc.

        vectors = self.vectors_of_segments.data
        segments = self.segments.data
        num_segments = np.shape(segments)[0]

        if self.intensities is None:
            print("The VDFSegment does not have the attribute intensities."
                  "All intensities will be set to ones.")
            intensities = np.ones_like(vectors)
            print(np.shape(intensities), 'np.shape(intensities)')
        else:
            intensities = self.intensities

        if sigma is None:
            sigma = calibration

        size_x, size_y = shape[0], shape[1]
        cx, cy = -size_x/2*calibration, -size_y/2*calibration
        X, Y = np.indices((size_x, size_y))
        X, Y = X * calibration + cx, Y * calibration + cy
        virtual_ed = np.zeros((size_x, size_y, num_segments))

        for i in range(num_segments):
            virtual_ed[..., i] = sum(list(map(
                lambda a, xo, yo: get_gaussian2d(a, xo, yo, x=X, y=Y,
                                               sigma=sigma),
                intensities[i], vectors[i][:,0], vectors[i][:,1])))

        virtual_ed = ElectronDiffraction(virtual_ed.T)

        return virtual_ed
