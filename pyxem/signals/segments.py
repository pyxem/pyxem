# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

"""Signal classes for nanocrystal segmentation results obtained from
i) machine learning (LearningSegment) and ii) VDF images (VDFSegment).

"""

import numpy as np
from tqdm import tqdm

from hyperspy.signals import Signal2D

from pyxem.signals import DiffractionVectors, ElectronDiffraction2D
from pyxem.utils.signal import transfer_signal_axes
from pyxem.utils.segment_utils import (
    norm_cross_corr,
    separate_watershed,
    get_gaussian2d,
)


class LearningSegment:
    _signal_type = "learning_segment"

    def __init__(self, factors, loadings):
        # Factors as Signal2D
        self.factors = factors
        # Corresponding loadings as Signal2D
        self.loadings = loadings

    def get_ncc_matrix(self):
        """Get the normalised correlation coefficient (NCC) matrix containing
        the NCC between each pair of segments.

        Returns
        -------
        ncc_matrix : Signal2D
            Normalised correlation coefficient matrix for loadings and factors.
        """
        # Set up empty matrices of correct size to store NCC values.
        num_comp = np.shape(self.loadings.data)[0]
        ncc_loadings = np.zeros((num_comp, num_comp))
        ncc_factors = np.zeros((num_comp, num_comp))
        factors = self.factors.map(np.nan_to_num, inplace=False).copy()
        loadings = self.loadings.map(np.nan_to_num, inplace=False).copy()
        # Iterate through loadings calculating NCC values.
        for i in np.arange(num_comp):
            ncc_loadings[i] = list(
                map(
                    lambda x: norm_cross_corr(x, template=loadings.data[i]),
                    loadings.data,
                )
            )
        # Iterate through factors calculating NCC values.
        for i in np.arange(num_comp):
            ncc_factors[i] = list(
                map(
                    lambda x: norm_cross_corr(x, template=factors.data[i]), factors.data
                )
            )
        # Convert matrix to Signal2D and set axes
        ncc_sig = Signal2D(np.array((ncc_loadings, ncc_factors)))
        ncc_sig.axes_manager.signal_axes[0].name = "index"
        ncc_sig.axes_manager.signal_axes[1].name = "index"
        ncc_sig.metadata.General.title = "Normalised Correlation Coefficient"

        return ncc_sig

    def correlate_learning_segments(self, corr_th_factors=0.4, corr_th_loadings=0.4):
        """Iterates through the factors and loadings and calculates the
        normalized cross-correlation between all factors and all
        loadings. Factors and loadings are summed if the correlations of
        both factors and loadings exceed the given thresholds.

        Parameters
        ----------
        corr_th_factors : float
            Correlation threshold for factors. Must be between -1 and 1.
            Factors and loadings are summed if both factors and loadings
            have normalized cross-correlations above corr_th_factors and
            corr_th_loadings, respectively.
        corr_th_loadings : int, optional
            Correlation threshold for loadings. Must be between -1 and 1.

        Returns
        -------
        learning_segment : LearningSegment
            LearningSegment where possibly some factors and loadings
            have been summed.
        """
        # If a mask was used during the decomposition, the factors and/or
        # loadings will contain nan, which must be converted to numbers prior
        # to the correlations calculations.
        factors = self.factors.map(np.nan_to_num, inplace=False)
        loadings = self.loadings.map(np.nan_to_num, inplace=False)
        factors = factors.copy().data
        loadings = loadings.copy().data
        correlated_loadings = np.zeros_like(loadings[:1])
        correlated_factors = np.zeros_like(factors[:1])

        # For each loading and factor, calculate the normalized
        # cross-correlation to all other loadings and factors, and define
        # add_indices for those with a value above corr_th_loadings and
        # corr_th_factors respectively.
        while np.shape(loadings)[0] > 0:
            corr_list_loadings = list(
                map(lambda x: norm_cross_corr(x, template=loadings[0]), loadings)
            )
            corr_list_factors = list(
                map(lambda x: norm_cross_corr(x, template=factors[0]), factors)
            )

            add_indices = np.where(
                list(
                    map(
                        lambda l, f: (l > corr_th_loadings and f > corr_th_factors),
                        corr_list_loadings,
                        corr_list_factors,
                    )
                )
            )

            correlated_loadings = np.append(
                correlated_loadings,
                np.array([np.sum(loadings[add_indices], axis=0)]),
                axis=0,
            )
            correlated_factors = np.append(
                correlated_factors,
                np.array([np.sum(factors[add_indices], axis=0)]),
                axis=0,
            )

            loadings = np.delete(loadings, add_indices, axis=0)
            factors = np.delete(factors, add_indices, axis=0)

        correlated_loadings = Signal2D(np.delete(correlated_loadings, 0, axis=0))
        correlated_factors = Signal2D(np.delete(correlated_factors, 0, axis=0))
        learning_segment = LearningSegment(
            factors=correlated_factors, loadings=correlated_loadings
        )
        return learning_segment

    def separate_learning_segments(
        self,
        min_intensity_threshold=0,
        min_distance=2,
        min_size=10,
        max_size=np.inf,
        max_number_of_grains=np.inf,
        marker_radius=2,
        threshold=False,
        exclude_border=False,
    ):
        """Segmentation of loading maps by the watershed
        segmentation method implemented in scikit-image [1,2].

        Parameters
        ----------
        min_intensity_threshold : float
            Loading segments with a maximum intensity below
            min_intensity_threshold are discarded.
        min_distance: int
            Minimum distance (in pixels) between grains required for
            them to be considered as separate grains.
        min_size : float
            Grains with size (i.e. total number of pixels) below
            min_size are discarded.
        max_size : float
            Grains with size (i.e. total number of pixels) above
            max_size are discarded.
        max_number_of_grains : int
            Maximum number of grains included in the returned separated
            grains. If it is exceeded, those with highest peak
            intensities will be returned.
        marker_radius : float
            If 1 or larger, each marker for watershed is expanded to a disk
            of radius marker_radius. marker_radius should not exceed
            2*min_distance.
        threshold: bool
            If True, a mask is calculated by thresholding the VDF image
            by the Li threshold method in scikit-image. If False
            (default), the mask is the boolean VDF image.
        exclude_border : int or True, optional
            If non-zero integer, peaks within a distance of
            exclude_border from the boarder will be discarded. If True,
            peaks at or closer than min_distance of the boarder, will be
            discarded.

        References
        ----------
        [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/
            plot_watershed.html
        [2] http://scikit-image.org/docs/dev/auto_examples/xx_applications/
            plot_coins_segmentation.html#sphx-glr-auto-examples-xx-
            applications-plot-coins-segmentation-py

        Returns
        -------
        learning_segment : LearningSegment
            LearningSegment where the loadings have been segmented and some
            factors have been repeated according to the new number of loading
            segments.
        """

        factors = self.factors.copy()
        loadings = self.loadings.copy()
        loadings_shape_x = loadings.data.shape[1]
        loadings_shape_y = loadings.data.shape[2]
        factors_shape_x = factors.data.shape[1]
        factors_shape_y = factors.data.shape[2]

        loadings_segments = np.array(
            loadings.map(
                separate_watershed,
                show_progressbar=True,
                inplace=False,
                min_distance=min_distance,
                min_size=min_size,
                max_size=max_size,
                max_number_of_grains=max_number_of_grains,
                marker_radius=marker_radius,
                threshold=threshold,
                exclude_border=exclude_border,
            ),
            dtype=object,
        )

        segments, factors_of_segments = [], []
        num_segs_tot = 0
        for loading_segment, factor in zip(loadings_segments, factors):
            segments = np.append(segments, loading_segment)
            num_segs = np.shape(loading_segment)[0]
            factors_of_segments = np.append(
                factors_of_segments,
                np.broadcast_to(factor, (num_segs, factors_shape_x, factors_shape_y)),
            )
            num_segs_tot += num_segs
        segments = segments.reshape((num_segs_tot, loadings_shape_y, loadings_shape_x))
        factors_of_segments = factors_of_segments.reshape(
            (num_segs_tot, factors_shape_x, factors_shape_y)
        )

        delete_indices = list(
            map(lambda x: x.max() < min_intensity_threshold, segments)
        )
        delete_indices = np.where(delete_indices)
        segments = np.delete(segments, delete_indices, axis=0)
        factors_of_segments = np.delete(factors_of_segments, delete_indices, axis=0)

        # if TraitError is raised, it is likely no segements were found
        segments = Signal2D(segments).transpose(navigation_axes=[0], signal_axes=[2, 1])
        factors_of_segments = Signal2D(factors_of_segments)
        learning_segment = LearningSegment(
            factors=factors_of_segments, loadings=segments
        )
        return learning_segment


class VDFSegment:
    _signal_type = "vdf_segment"

    def __init__(self, segments, vectors_of_segments, intensities=None):
        # Segments as Signal2D
        self.segments = segments
        # DiffractionVectors
        self.vectors_of_segments = vectors_of_segments
        # Intensities corresponding to each vector
        self.intensities = intensities

    def get_ncc_matrix(self):
        """Get the normalised correlation coefficient (NCC) matrix containing
        the NCC between each pair of segments.

        Returns
        -------
        ncc_matrix : Signal2D
            Normalised correlation coefficient matrix.
        """
        # TODO: This code should be factored out for reuse in other ncc method.
        # Set up an empty matrix of correct size to store NCC values.
        num_comp = np.shape(self.segments.data)[0]
        ncc_matrix = np.zeros((num_comp, num_comp))
        # Iterate through segments calculating NCC values.
        for i in np.arange(num_comp):
            ncc_matrix[i] = list(
                map(
                    lambda x: norm_cross_corr(x, template=self.segments.data[i]),
                    self.segments.data,
                )
            )
        # Convert matrix to Signal2D and set axes
        ncc_sig = Signal2D(ncc_matrix)
        ncc_sig.axes_manager.signal_axes[0].name = "segment index"
        ncc_sig.axes_manager.signal_axes[1].name = "segment index"
        ncc_sig.metadata.General.title = "Normalised Correlation Coefficient"
        return ncc_sig

    def correlate_vdf_segments(
        self, corr_threshold=0.7, vector_threshold=4, segment_threshold=3
    ):
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

        if segment_threshold > vector_threshold:
            raise ValueError(
                "segment_threshold must be smaller than or "
                "equal to vector_threshold."
            )

        segments = self.segments.data.copy()
        num_vectors = np.shape(vectors)[0]
        gvectors = np.array(np.empty(num_vectors, dtype=object))
        vector_indices = np.array(np.empty(num_vectors, dtype=object))

        for i in np.arange(num_vectors):
            gvectors[i] = np.array(vectors[i].copy())
            vector_indices[i] = np.array([i], dtype=int)

        correlated_segments = np.zeros_like(segments[:1])
        correlated_vectors = np.array([0.0], dtype=object)
        correlated_vectors[0] = np.array(np.zeros_like(vectors[:1]))
        correlated_vector_indices = np.array([0], dtype=object)
        correlated_vector_indices[0] = np.array([0])
        i = 0
        pbar = tqdm(total=np.shape(segments)[0])
        while np.shape(segments)[0] > i:
            # For each segment, calculate the normalized cross-correlation to
            # all other segments, and define add_indices for those with a value
            # above corr_threshold.
            corr_list = list(
                map(lambda x: norm_cross_corr(x, template=segments[i]), segments)
            )

            corr_add = list(map(lambda x: x > corr_threshold, corr_list))
            add_indices = np.where(corr_add)
            # If there are more add_indices than vector_threshold,
            # sum segments and add their vectors. Otherwise, discard segment.
            if (
                np.shape(add_indices[0])[0] >= vector_threshold
                and np.shape(add_indices[0])[0] > 1
            ):
                new_segment = np.array([np.sum(segments[add_indices], axis=0)])
                if segment_threshold > 1:
                    segment_check = np.zeros_like(segments[add_indices], dtype=int)
                    segment_check[np.where(segments[add_indices])] = 1
                    segment_check = np.sum(segment_check, axis=0, dtype=int)
                    segment_mask = np.zeros_like(segments[0], dtype=bool)
                    segment_mask[np.where(segment_check >= segment_threshold)] = 1
                    new_segment = new_segment * segment_mask
                correlated_segments = np.append(
                    correlated_segments, new_segment, axis=0
                )
                add_indices = add_indices[0]
                new_vectors = np.array([0], dtype=object)
                new_vectors[0] = np.concatenate(gvectors[add_indices], axis=0).reshape(
                    -1, 2
                )
                correlated_vectors = np.append(correlated_vectors, new_vectors, axis=0)
                new_indices = np.array([0], dtype=object)
                new_indices[0] = np.concatenate(
                    vector_indices[add_indices], axis=0
                ).reshape(-1, 1)
                correlated_vector_indices = np.append(
                    correlated_vector_indices, new_indices, axis=0
                )
            elif np.shape(add_indices[0])[0] >= vector_threshold:
                add_indices = add_indices[0]
                correlated_segments = np.append(
                    correlated_segments, segments[add_indices], axis=0
                )
                correlated_vectors = np.append(
                    correlated_vectors, gvectors[add_indices], axis=0
                )
                correlated_vector_indices = np.append(
                    correlated_vector_indices, vector_indices[add_indices], axis=0
                )
            else:
                add_indices = i
            segments = np.delete(segments, add_indices, axis=0)
            gvectors = np.delete(gvectors, add_indices, axis=0)
            vector_indices = np.delete(vector_indices, add_indices, axis=0)

        pbar.close()
        correlated_segments = np.delete(correlated_segments, 0, axis=0)
        correlated_vectors = np.delete(correlated_vectors, 0, axis=0)
        correlated_vector_indices = np.delete(correlated_vector_indices, 0, axis=0)
        correlated_vector_intensities = np.array(
            np.empty(len(correlated_vectors)), dtype=object
        )

        # Sum the intensities in the original segments and assign those to the
        # correct vectors by referring to vector_indices.
        # If segment_mask has been used, use the segments as masks too.
        if segment_threshold > 1:
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(
                    len(correlated_vector_indices[i])
                )
                segment_mask = np.zeros_like(segment_mask)
                segment_mask[np.where(correlated_segments[i])] = 1
                segment_intensities = np.sum(
                    self.segments.data * segment_mask, axis=(1, 2)
                )
                for n, index in zip(
                    range(len(correlated_vector_indices[i])),
                    correlated_vector_indices[i],
                ):
                    correlated_vector_intensities[i][n] = np.sum(
                        segment_intensities[index]
                    )
        else:
            segment_intensities = np.sum(self.segments.data, axis=(1, 2))
            for i in range(len(correlated_vectors)):
                correlated_vector_intensities[i] = np.zeros(
                    len(correlated_vector_indices[i])
                )
                for n, index in zip(
                    range(len(correlated_vector_indices[i])),
                    correlated_vector_indices[i],
                ):
                    correlated_vector_intensities[i][n] = np.sum(
                        segment_intensities[index]
                    )

        vdfseg = VDFSegment(
            Signal2D(correlated_segments),
            DiffractionVectors(correlated_vectors),
            correlated_vector_intensities,
        )

        # Transfer axes properties of segments
        vdfseg.segments = transfer_signal_axes(vdfseg.segments, self.segments)
        n = vdfseg.segments.axes_manager.navigation_axes[0]
        n.name = "n"
        n.units = "number"

        return vdfseg

    def get_virtual_electron_diffraction(self, calibration, shape, sigma):
        """Obtain a virtual electron diffraction signal that consists
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
            per pixel. 'calibration' is a decent starting value.

        Returns
        -------
        virtual_ed : ElectronDiffraction2D
            Virtual electron diffraction signal consisting of one
            virtual diffraction pattern for each segment.
        """
        vectors = self.vectors_of_segments.data
        segments = self.segments.data
        num_segments = np.shape(segments)[0]

        if self.intensities is None:
            raise ValueError(
                "The VDFSegment does not have the attribute "
                "intensities, required for this method."
            )
        else:
            intensities = self.intensities

        # TODO: Refactor this to use the diffsims simulation to plot functionality
        size_x, size_y = shape[0], shape[1]
        cx, cy = -size_x / 2 * calibration, -size_y / 2 * calibration
        x, y = np.indices((size_x, size_y))
        x, y = x * calibration + cx, y * calibration + cy
        virtual_ed = np.zeros((size_x, size_y, num_segments))

        for i in range(num_segments):
            # Allow plotting for segments that are only associated with
            # one vector.
            if np.shape(np.shape(vectors[i]))[0] <= 1:
                virtual_ed[..., i] = get_gaussian2d(
                    intensities[i],
                    vectors[i][..., 0],
                    vectors[i][..., 1],
                    x=x,
                    y=y,
                    sigma=sigma,
                )
            # Allow plotting for segments associated with several vectors.
            else:
                virtual_ed[..., i] = sum(
                    list(
                        map(
                            lambda a, xo, yo: get_gaussian2d(
                                a, xo, yo, x=x, y=y, sigma=sigma
                            ),
                            intensities[i],
                            vectors[i][..., 0],
                            vectors[i][..., 1],
                        )
                    )
                )

        virtual_ed = ElectronDiffraction2D(virtual_ed.T)

        return virtual_ed
