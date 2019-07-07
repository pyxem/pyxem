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

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

from hyperspy.signals import Signal2D, BaseSignal

from pyxem.utils.vdf_utils import (norm_cross_corr, get_vectors_and_indices_i,
                                   get_gaussian2d, get_circular_mask)
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.electron_diffraction import ElectronDiffraction
from pyxem.utils.sim_utils import transfer_signal_axes


class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vectors = None

    def get_virtual_decomposed_signal(self, loadings, calibration, dp_shape,
                                      sigma=None):
        """After having decomposed a VDFImage signal, obtain a virtual
        electron diffraction signal that consists of one virtual
        diffraction pattern for each decomposition component. Each
        virtual pattern is composed of Gaussians centered at each
        vector position with an amplitude given by the intensity of the
        corresponding loading pattern.

        Parameters
        ----------
        calibration : float
            Reciprocal space calibration in inverse Angstrom per pixel.
        loadings : BaseSignal
            Decomposition loadings with dimensions ( nc | ng ), where nc
            is the number of components and ng the number of vectors.
        dp_shape : tuple of int
            Shape of the diffraction patterns (dp_length_x, dp_length_y)
            in pixels, where dp_length_x and dp_length_y are integers.
        sigma : float
            The standard deviation of the Gaussians in inverse Angstrom
            per pixel. If None (default), sigma=calibration.

        Returns
        -------
        virtual_ed : ElectronDiffraction
            Virtual electron diffraction signal consisting of one
            virtual diffraction pattern for each decomposition component.
        """
        intensities = loadings.data
        vectors = self.vectors.data

        num_components = np.shape(intensities)[0]

        if sigma is None:
            sigma = calibration

        size_x, size_y = dp_shape[0], dp_shape[1]
        cx, cy = -size_x / 2 * calibration, -size_y / 2 * calibration
        x, y = np.indices((size_x, size_y))
        x, y = x * calibration + cx, y * calibration + cy
        virtual_ed = np.zeros((size_x, size_y, num_components))

        for i in range(num_components):
            virtual_ed[..., i] = sum(list(map(
                lambda a, xo, yo: get_gaussian2d(
                    a, xo, yo, x=x, y=y, sigma=sigma),
                intensities[i], vectors[..., 0], vectors[..., 1])))

        virtual_ed = ElectronDiffraction(virtual_ed.T)

        return virtual_ed


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
        vdfseg : VDFSegment
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
            corr_add = list(map(lambda x: x > corr_threshold, corr_list))

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
            gvector_intensities[i] = np.zeros(len(vector_indices[i]))
            for n, index in zip(range(len(vector_indices[i])),
                                vector_indices[i]):
                gvector_intensities[i][n] = np.sum(segment_intensities[index])

        vdfseg = VDFSegment(Signal2D(image_stack), DiffractionVectors(gvectors),
                            gvector_intensities)

        # Transfer axes properties of segments
        vdfseg.segments = transfer_signal_axes(vdfseg.segments, self.segments)
        n = vdfseg.segments.axes_manager.navigation_axes[0]
        n.name = 'n'
        n.units = 'number'

        return vdfseg

    def threshold_segments(self, min_intensity_threshold=None,
                           vector_number_threshold=None):
        """Obtain a VDFSegment where the segments with a lower number of
        vectors than that defined by vector_number_threshold or with a
        smaller maximum intensity than that defined by
        min_intensity_threshold have been removed.

        Parameters
        ----------
        min_intensity_threshold : int
            Threshold for the minimum intensity (in a single pixel) in a
            segment for it to not be discarded.
        vector_number_threshold : int
            Threshold for the minimum number of vectors a segment should
            contain for it to not be removed.

        Returns
        -------
        vdfseg : VDFSegment
            As input, except that segments might have been removed after
            thresholds based on min intensity and/or number of vectors.
        """
        if min_intensity_threshold is None and vector_number_threshold is None:
            raise ValueError("Specify input threshold.")

        image_stack = self.segments.data.copy()
        vectors = self.vectors_of_segments.data.copy()
        intensities = self.intensities.copy()

        if min_intensity_threshold is not None:
            n = 0
            while np.shape(image_stack)[0] > n:
                if np.max(image_stack[n]) < min_intensity_threshold:
                    image_stack = np.delete(image_stack, n, axis=0)
                    vectors = np.delete(vectors, n, axis=0)
                    intensities = np.delete(intensities, n, axis=0)
                else:
                    n = n + 1

        if vector_number_threshold is not None:
            n = 0
            while np.shape(image_stack)[0] > n:
                if len(np.shape(vectors[n])) == 2 and \
                        np.shape(vectors[n])[0] >= vector_number_threshold:
                    n = n + 1
                else:
                    image_stack = np.delete(image_stack, n, axis=0)
                    vectors = np.delete(vectors, n, axis=0)
                    intensities = np.delete(intensities, n, axis=0)

        if not np.any(image_stack):
            print('No segments left. Check the input thresholds.')
            return 0

        vdfseg = VDFSegment(Signal2D(image_stack), DiffractionVectors(vectors),
                            intensities)
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

        virtual_ed = ElectronDiffraction(virtual_ed.T)

        return virtual_ed

    def get_separated_electron_diffraction_signals(self, original_signal,
                                                   radius):
        """Obtain electron diffraction signals based on an original
        electron diffraction signal and its separated vdf segments and
        vectors, so that each output signal contains segments that do
        not overlap. The intensities of each vector in the output
        signals are collected directly from the original signal inside a
        circle defined by radius.

        Parameters
        ----------
        original_signal : ElectronDiffraction
            Electron diffraction signal that the VDFSegment originates
            from.
        radius : float
            Radius of the integration window in reciprocal Angstroms.
            Similar to the radius used for VDF images.

        Returns
        -------
        separated_signals : list of ElectronDiffraction
            A list of electron diffraction signals corresponding to the
            original signal, but where overlapping grains are found in
            separate signals. The number of signals will reflect the max
            number of grains that overlap with one grain.
        """
        vectors = self.vectors_of_segments.data
        segments = self.segments.data
        num_segments = np.shape(segments)[0]

        segment_masks = (segments > 0).choose(segments, 1.).astype('int')
        tot_segment_masks_sum = np.sum(segment_masks, axis=0)

        assigned_num = np.zeros(num_segments)
        segment_masks_sum = [np.zeros_like(tot_segment_masks_sum)]
        for i, segment_mask in zip(range(num_segments), segment_masks):
            if np.max(tot_segment_masks_sum[np.where(segment_mask)]) > 1:
                n, finished = 0, False
                while n < np.shape(segment_masks_sum)[0] and finished is False:
                    if np.max(segment_masks_sum[n] + segment_mask) > 1:
                        n = n+1
                    else:
                        finished = True
                assigned_num[i] = n
                if n+1 > np.shape(segment_masks_sum)[0]:
                    segment_masks_sum.append(np.zeros_like(
                        tot_segment_masks_sum))
                segment_masks_sum[n] += segment_mask
            else:
                assigned_num[i] = 0
                segment_masks_sum[0] += segment_mask

        dp_shape_y, dp_shape_x = original_signal.axes_manager.signal_shape
        nav_shape_y, nav_shape_x = original_signal.axes_manager.navigation_shape
        scale_y = original_signal.axes_manager.signal_axes[0].scale
        scale_x = original_signal.axes_manager.signal_axes[1].scale
        cy = original_signal.axes_manager.signal_axes[0].offset
        cx = original_signal.axes_manager.signal_axes[1].offset

        y, x = np.indices((dp_shape_x, dp_shape_y))
        x, y = x*scale_x, y*scale_y

        separated_signals = []
        for j in range(int(np.max(assigned_num)+1)):
            segment_masks_j = segment_masks[np.where(assigned_num == j)]
            vectors_j = vectors[np.where(assigned_num == j)]
            signal_j = np.zeros((nav_shape_x, nav_shape_y, dp_shape_x,
                                 dp_shape_y))
            for i in range(np.shape(vectors_j)[0]):
                if np.shape(np.shape(vectors_j[i]))[0] <= 1:
                    vector_mask_i = get_circular_mask(vectors_j[i], radius,
                                                      cx, cy, x, y)
                else:
                    vector_mask_i = np.sum(list(map(
                        lambda b: get_circular_mask(b, radius, cx, cy, x, y),
                        vectors_j[i])), axis=0)
                signal_j[np.where(segment_masks_j[i])] = \
                    original_signal.data[np.where(segment_masks_j[i])] \
                    * vector_mask_i
            separated_signals.append(ElectronDiffraction(signal_j))

        return separated_signals

    def plot_all_segments(self, ved=None, image_to_plot_segments_on=None,
                          image_to_plot_ved_on=None):
        """Plot all segments in one figure, where each segment has a
        distinct color. Optionally also plot the virtual electron
        diffraction signal with colors corresponding to those of the
        segments. The colors are uniformly distributed within the
        matplotlib colormap 'gist_rainbow', excluding the lowest and
        highest values (black and white).

        Parameters
        ----------
        ved : ElectronDiffraction
            Electron diffraction signal that is the virtual electron
            diffraction signal corresponding to the VDF segments.
        image_to_plot_segments_on : np.array
            If provided, the segments will be plotted on top of this
            image.
        image_to_plot_ved_on : np.array
            If provided, ved will be plotted on top of this image.
        Returns
        -------
        None
        """
        cmap = get_cmap('gist_rainbow')
        N = 256
        segs = self.segments
        nu = segs.axes_manager.navigation_size
        step = N / (nu + 2)
        steps = np.arange(step, step * (nu + 1), step) / N

        plt.figure()
        if image_to_plot_segments_on is not None:
            plt.imshow(image_to_plot_segments_on, cmap='gray', alpha=0.9)
        for i in np.arange(nu):
            a = cmap(steps[i])
            vals = np.zeros((N, 4))
            vals[:, 0] += a[0]
            vals[:, 1] += a[1]
            vals[:, 2] += a[2]
            vals[:, 3] = np.linspace(0., 1., N)
            cmap_i = ListedColormap(vals)
            plt.imshow(segs.inav[i].data / segs.inav[i].data.max(),
                       cmap=cmap_i)
        plt.title('Segments')
        plt.show()

        if ved is not None:
            plt.figure()
            if image_to_plot_ved_on is not None:
                plt.imshow(image_to_plot_ved_on, cmap='gray', alpha=0.9)
            for i in np.arange(nu):
                a = cmap(steps[i])
                vals = np.zeros((N, 4))
                vals[:, 0] += a[0]
                vals[:, 1] += a[1]
                vals[:, 2] += a[2]
                vals[:, 3] = np.linspace(0., 1., N)
                cmap_i = ListedColormap(vals)
                plt.imshow(ved.inav[i].data / ved.inav[i].data.max(),
                           cmap=cmap_i)
            plt.title('Virtual signal')
            plt.show()

        return None
