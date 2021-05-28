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
"""Signal class for virtual diffraction contrast images."""
import numpy as np

from hyperspy.signals import Signal2D

from pyxem.signals import DiffractionVectors, VDFSegment
from pyxem.utils.signal import transfer_signal_axes
from pyxem.utils.segment_utils import separate_watershed


class VirtualDarkFieldImage(Signal2D):
    _signal_type = "virtual_dark_field"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = None

    def get_vdf_segments(
        self,
        min_distance=1,
        min_size=1,
        max_size=np.inf,
        max_number_of_grains=np.inf,
        marker_radius=1,
        threshold=False,
        exclude_border=False,
    ):
        """Separate segments from each of the virtual dark field (VDF) images
        using edge-detection by the Sobel transform and the watershed
        segmentation method implemented in scikit-image [1,2]. Obtain a
        VDFSegment, similar to VDFImage, but where each image is a
        segment of a VDF and the vectors correspond to each segment and
        are not necessarily unique.

        Parameters
        ----------
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
        vdfsegs : VDFSegment
            VDFSegment object containing segments (i.e. grains) of
            single virtual dark field images with corresponding vectors.
        """
        vdfs = self.copy()
        vectors = self.vectors.data

        # TODO : Add aperture radius as an attribute of VDFImage?

        # Create an array of length equal to the number of vectors where each
        # element is a object with shape (n: number of segments for this
        # VDFImage, VDFImage size x, VDFImage size y).
        vdfsegs = np.array(
            vdfs.map(
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

        segments, vectors_of_segments = [], []
        for i, vector in zip(np.arange(vectors.size), vectors):
            segments = np.append(segments, vdfsegs[i])
            num_segs = np.shape(vdfsegs[i])[0]
            vectors_of_segments = np.append(
                vectors_of_segments, np.broadcast_to(vector, (num_segs, 2))
            )

        vectors_of_segments = vectors_of_segments.reshape((-1, 2))
        segments = segments.reshape(
            (
                np.shape(vectors_of_segments)[0],
                vdfs.axes_manager.signal_shape[0],
                vdfs.axes_manager.signal_shape[1],
            )
        )
        # Calculate the total intensities of each segment
        segment_intensities = np.array(
            [[np.sum(x, axis=(0, 1))] for x in segments], dtype="object"
        )

        # if TraitError is raised, it is likely no segments were found
        segments = Signal2D(segments).transpose(navigation_axes=[0], signal_axes=[2, 1])
        # Create VDFSegment and transfer axes calibrations
        vdfsegs = VDFSegment(
            segments, DiffractionVectors(vectors_of_segments), segment_intensities
        )
        vdfsegs.segments = transfer_signal_axes(vdfsegs.segments, vdfs)
        n = vdfsegs.segments.axes_manager.navigation_axes[0]
        n.name = "n"
        n.units = "number"
        vdfsegs.vectors_of_segments.axes_manager.set_signal_dimension(1)
        vdfsegs.vectors_of_segments = transfer_signal_axes(
            vdfsegs.vectors_of_segments, self.vectors
        )
        n = vdfsegs.vectors_of_segments.axes_manager.navigation_axes[0]
        n.name = "n"
        n.units = "number"

        return vdfsegs
