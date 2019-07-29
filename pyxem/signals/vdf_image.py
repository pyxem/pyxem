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
"""Signal class for virtual diffraction contrast images and segments.

"""

from pyxem.signals import push_metadata_through

import numpy as np
from tqdm import tqdm

from hyperspy.signals import Signal2D

from pyxem.utils.vdf_utils import norm_cross_corr, get_gaussian2d
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals import transfer_signal_axes


class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vectors = None


class VDFSegment:
    _signal_type = "vdf_segment"

    def __init__(self, segments, vectors_of_segments, intensities=None, *args,
                 **kwargs):
        # Segments as Signal2D
        self.segments = segments
        # DiffractionVectors
        self.vectors_of_segments = vectors_of_segments
        # Intensities corresponding to each vector
        self.intensities = intensities

    def correlate_segments(self, corr_threshold=0.7, vector_threshold=4,
                           segment_threshold=3):
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
        vector_threshold : int, optional
            Correlated segments having a number of vectors less than
            vector_threshold will be discarded.
        segment_threshold : int, optional
            Correlated segment intensities that lie in a region where
            a number of segments less than segment_thresholdhave been
            found, are set to 0, i.e. the resulting segment will only
            have intensities above 0 where at least a number of
            segment_threshold segments have intensitives above 0.

        Returns
        -------
        vdfseg : VDFSegment
            The VDFSegment instance updated according to the image
            correlation results.
        """
        vectors = self.vectors_of_segments.data
        if len(np.shape(vectors)) <= 1:
            raise ValueError("Input vectors are not of correct shape. Try to "
                             "rerun correlate_segments on original VDFSegment "
                             "resulting from get_vdf_segments.")
        if segment_threshold > vector_threshold:
            raise ValueError("segment_threshold must be smaller than or "
                             "equal to vector_threshold.")

        segments = self.segments.data.copy()
        num_vectors = np.shape(vectors)[0]
        gvectors = np.array(np.empty(num_vectors, dtype=object))
        vector_indices = np.array(np.empty(num_vectors, dtype=object))

        for i in np.arange(num_vectors):
            gvectors[i] = np.array(vectors[i].copy())
            vector_indices[i] = np.array([i], dtype=int)

        correlated_segments = np.zeros_like(segments[:1])
        correlated_vectors = np.array([0.], dtype=object)
        correlated_vectors[0] = np.array(np.zeros_like(vectors[:1]))
        correlated_vector_indices = np.array([0], dtype=object)
        correlated_vector_indices[0] = np.array([0])
        i = 0
        pbar = tqdm(total=np.shape(segments)[0])
        while np.shape(segments)[0] > i:
            # For each segment, calculate the normalized cross-correlation to
            # all other segments, and define add_indices for those with a value
            # above corr_threshold.
            corr_list = list(map(
                lambda x: norm_cross_corr(x, template=segments[i]),
                segments))

            corr_add = list(map(lambda x: x > corr_threshold, corr_list))
            add_indices = np.where(corr_add)
            # If there are more add_indices than vector_threshold,
            # sum segments and add their vectors. Otherwise, discard segment.
            if (np.shape(add_indices[0])[0] >= vector_threshold and
                    np.shape(add_indices[0])[0] > 1):
                new_segment = np.array([np.sum(segments[add_indices],
                                               axis=0)])
                if segment_threshold > 1:
                    segment_check = np.zeros_like(segments[add_indices],
                                                  dtype=int)
                    segment_check[np.where(segments[add_indices])] = 1
                    segment_check = np.sum(segment_check, axis=0, dtype=int)
                    segment_mask = np.zeros_like(segments[0], dtype=bool)
                    segment_mask[
                        np.where(segment_check >= segment_threshold)] = 1
                    new_segment = new_segment * segment_mask
                correlated_segments = np.append(correlated_segments,
                                                new_segment, axis=0)
                add_indices = add_indices[0]
                new_vectors = np.array([0], dtype=object)
                new_vectors[0] = np.concatenate(gvectors[add_indices],
                                                axis=0).reshape(-1, 2)
                correlated_vectors = np.append(correlated_vectors,
                                               new_vectors, axis=0)
                new_indices = np.array([0], dtype=object)
                new_indices[0] = np.concatenate(vector_indices[add_indices],
                                                axis=0).reshape(-1, 1)
                correlated_vector_indices = np.append(correlated_vector_indices,
                                                      new_indices, axis=0)
            elif np.shape(add_indices[0])[0] >= vector_threshold:
                add_indices = add_indices[0]
                correlated_segments = np.append(correlated_segments,
                                                segments[add_indices],
                                                axis=0)
                correlated_vectors = np.append(correlated_vectors,
                                               gvectors[add_indices], axis=0)
                correlated_vector_indices = np.append(correlated_vector_indices,
                                                      vector_indices[
                                                          add_indices],
                                                      axis=0)
            else:
                add_indices = i
            segments = np.delete(segments, add_indices, axis=0)
            gvectors = np.delete(gvectors, add_indices, axis=0)
            vector_indices = np.delete(vector_indices, add_indices, axis=0)

        pbar.close()
        correlated_segments = np.delete(correlated_segments, 0, axis=0)
        correlated_vectors = np.delete(correlated_vectors, 0, axis=0)
        correlated_vector_indices = np.delete(correlated_vector_indices,
                                              0, axis=0)
        correlated_vector_intensities = np.array(
            np.empty(len(correlated_vectors)),
            dtype=object)

        # Sum the intensities in the original segments and assign those to the
        # correct vectors by referring to vector_indices.
        # If segment_mask has been used, use the segments as masks too.
        if segment_threshold > 1:
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(len(
                    correlated_vector_indices[i]))
                segment_mask = np.zeros_like(segment_mask)
                segment_mask[np.where(correlated_segments[i])] = 1
                segment_intensities = np.sum(self.segments.data * segment_mask,
                                             axis=(1, 2))
                for n, index in zip(range(len(correlated_vector_indices[i])),
                                    correlated_vector_indices[i]):
                    correlated_vector_intensities[i][n] = np.sum(
                        segment_intensities[index])
        else:
            segment_intensities = np.sum(self.segments.data, axis=(1, 2))
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(
                    len(correlated_vector_indices[i]))
                for n, index in zip(range(len(correlated_vector_indices[i])),
                                    correlated_vector_indices[i]):
                    correlated_vector_intensities[i][n] = np.sum(
                        segment_intensities[index])

        vdfseg = VDFSegment(Signal2D(correlated_segments),
                            DiffractionVectors(correlated_vectors),
                            correlated_vector_intensities)

        # Transfer axes properties of segments
        vdfseg.segments = transfer_signal_axes(vdfseg.segments, self.segments)
        n = vdfseg.segments.axes_manager.navigation_axes[0]
        n.name = 'n'
        n.units = 'number'

        return vdfseg

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
        vectors = self.vectors_of_segments.data
        segments = self.segments.data
        num_segments = np.shape(segments)[0]

        if self.intensities is None:
            print("The VDFSegment does not have the attribute intensities."
                  "All intensities will be set to ones.")
            intensities = np.ones_like(vectors)
        else:
            intensities = self.intensities

        if sigma is None:
            sigma = calibration

        size_x, size_y = shape[0], shape[1]
        cx, cy = -size_x/2*calibration, -size_y/2*calibration
        x, y = np.indices((size_x, size_y))
        x, y = x * calibration + cx, y * calibration + cy
        virtual_ed = np.zeros((size_x, size_y, num_segments))

        for i in range(num_segments):
            if np.shape(np.shape(vectors[i]))[0] <= 1:
                virtual_ed[..., i] = get_gaussian2d(
                    intensities[i], vectors[i][..., 0],
                    vectors[i][..., 1], x=x, y=y, sigma=sigma)
            else:
                virtual_ed[..., i] = sum(list(map(
                    lambda a, xo, yo: get_gaussian2d(
                        a, xo, yo, x=x, y=y, sigma=sigma),
                    intensities[i], vectors[i][..., 0], vectors[i][..., 1])))

        virtual_ed = ElectronDiffraction2D(virtual_ed.T)

        return virtual_ed
