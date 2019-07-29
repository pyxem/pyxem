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

"""VDF generator, VDF segment generator and associated tools.

"""

from pyxem.signals.vdf_image import VDFImage, VDFSegment
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.vdf_utils import (normalize_vdf, separate)
from pyxem.utils.sim_utils import (transfer_signal_axes,
                                   transfer_navigation_axes_to_signal_axes)

from hyperspy.api import roi
from hyperspy.signals import BaseSignal, Signal2D
import numpy as np


class VDFGenerator:
    """Generates VDF images for a specified signal and set of aperture
    positions.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.
    vectors: DiffractionVectors(optional)
        The vector positions, in calibrated units, at which to position
        integration windows for VDF formation.

    """

    def __init__(self, signal, vectors=None, *args, **kwargs):
        # If ragged the signal axes will not be defined

        if vectors is None:
            unique_vectors = None
        elif len(vectors.axes_manager.signal_axes) == 0:
            unique_vectors = vectors.get_unique_vectors(*args, **kwargs)
        else:
            unique_vectors = vectors

        self.signal = signal
        self.vectors = unique_vectors
        self.vectors = transfer_signal_axes(self.vectors, self.signal)

    def get_vector_vdf_images(self, radius, normalize=False):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction2D Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        radius : float
            Radius of the integration window in reciprocal angstroms.

        normalize : boolean
            If True each VDF image is normalized so that the maximum intensity
            in each VDF is 1.

        Returns
        -------
        vdfs : VDFImage
            VDFImage object containing virtual dark field images for all unique
            vectors.
        """
        if self.vectors:
            vdfs = []
            for v in self.vectors.data:
                disk = roi.CircleROI(cx=v[0], cy=v[1], r=radius, r_inner=0)
                vdf = disk(self.signal,
                           axes=self.signal.axes_manager.signal_axes)
                vdfs.append(vdf.sum((2, 3)).as_signal2D((0, 1)).data)

            vdfim = VDFImage(np.asarray(vdfs))

            if normalize is True:
                vdfim.map(normalize_vdf)

        else:
            raise ValueError("DiffractionVectors non-specified by user. Please "
                             "initialize VDFGenerator with some vectors. ")

        # Set calibration to same as signal
        vdfim = transfer_navigation_axes_to_signal_axes(vdfim, self.signal)

        # Assign vectors used to generate images to vdfim attribute.
        vdfim.vectors = self.vectors
        vdfim.vectors = transfer_signal_axes(vdfim.vectors, self.vectors)

        return vdfim

    def get_concentric_vdf_images(self, k_min, k_max, k_steps, normalize=False):
        """Obtain the intensity scattered at each navigation position in an
        ElectronDiffraction2D Signal by summation over a series of concentric
        in annuli between a specified inner and outer radius in a number of
        steps.

        Parameters
        ----------
        k_min : float
            Minimum radius of the annular integration window in reciprocal
            angstroms.

        k_max : float
            Maximum radius of the annular integration window in reciprocal
            angstroms.

        k_steps : int
            Number of steps within the annular integration window

        Returns
        -------
        vdfs : VDFImage
            VDFImage object containing virtual dark field images for all steps
            within the annulus.
        """
        k_step = (k_max - k_min) / k_steps
        k0s = np.linspace(k_min, k_max - k_step, k_steps)
        k1s = np.linspace(k_min + k_step, k_max, k_steps)

        ks = np.array((k0s, k1s)).T

        vdfs = []
        for k in ks:
            annulus = roi.CircleROI(cx=0, cy=0, r=k[1], r_inner=k[0])
            vdf = annulus(self.signal,
                          axes=self.signal.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2, 3)).as_signal2D((0, 1)).data)

        vdfim = VDFImage(np.asarray(vdfs))

        if normalize is True:
            vdfim.map(normalize_vdf)

        # Set calibration to same as signal
        vdfim = transfer_navigation_axes_to_signal_axes(vdfim, self.signal)

        return vdfim


class VDFSegmentGenerator:
    """Generates VDF segments for specified VDFImages and corresponding
    set of aperture positions (i.e. unique vectors).

    Parameters
    ----------
    vdfs : VDFImage
        The VDF images to be segmented.
    vectors: DiffractionVectors
        The vector positions corresponding to the VDF images.

    """

    def __init__(self, vdfs, *args, **kwargs):

        self.vdf_images = vdfs
        self.vectors = vdfs.vectors

    def get_vdf_segments(self, min_distance=1, min_size=10,
                         max_size=100, max_number_of_grains=np.inf,
                         threshold=False, exclude_border=False):
        """Separate segments (grains) from each of the VDF images using
        edge-detection by the sobel transform and the watershed
        segmentation method implemented in scikit-image [1,2]. Obtain a
        VDFSegment, similar to VDFImage, but where each image is a
        segment of a VDF and the vectors correspond to each segment and
        are not (necessarily) unique.

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
        vdfs = self.vdf_images
        vectors = self.vectors.data

        #TODO : Add aperture radius as an attribute of VDFImage and VDFSegment?

        # Create an array of length equal to the number of vectors where each
        # element is a np.object with shape (n: number of segments for this
        # VDFImage, VDFImage size x, VDFImage size y).
        # map(lambda ids: my_function(ids, ip), volume_ids);
        vdfsegs = np.array(vdfs.map(
            separate, show_progressbar=True, inplace=False,
            min_distance=min_distance, min_size=min_size, max_size=max_size,
            max_number_of_grains=max_number_of_grains, threshold=threshold,
            exclude_border=exclude_border), dtype=np.object)

        segments, vectors_of_segments = [], []
        for i, vector in zip(np.arange(vectors.size), vectors):
            segments = np.append(segments, vdfsegs[i])
            num_segs = np.shape(vdfsegs[i])[0]
            vectors_of_segments = np.append(vectors_of_segments,
                                            np.broadcast_to(vector,
                                                            (num_segs, 2)))

        vectors_of_segments = vectors_of_segments.reshape((-1, 2))
        segments = segments.reshape((np.shape(vectors_of_segments)[0],
                                     vdfs.axes_manager.signal_shape[0],
                                     vdfs.axes_manager.signal_shape[1]))

        segments = Signal2D(segments).transpose(navigation_axes=[0],
                                                signal_axes=[2, 1])

        # Create VDFSegment and transfer axes calibrations
        vdfsegs = VDFSegment(segments, DiffractionVectors(vectors_of_segments))
        vdfsegs.segments = transfer_signal_axes(vdfsegs.segments,
                                                self.vdf_images)
        n = vdfsegs.segments.axes_manager.navigation_axes[0]
        n.name = 'n'
        n.units = 'number'
        vdfsegs.vectors_of_segments.axes_manager.set_signal_dimension(1)
        vdfsegs.vectors_of_segments = transfer_signal_axes(
            vdfsegs.vectors_of_segments, self.vectors)
        n = vdfsegs.vectors_of_segments.axes_manager.navigation_axes[0]
        n.name = 'n'
        n.units = 'number'

        return vdfsegs
