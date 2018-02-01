# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

from hyperspy._signals.lazy import LazySignal
from hyperspy.api import interactive, stack
from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from pyxem.signals.diffraction_profile import DiffractionProfile
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import *
from pyxem.utils.peakfinders2D import *


def separate(VDF_temp,
             min_distance = 3,
             threshold = 0.4,
             discard_size = 3,
             plot_on = False):
    """Separate grains from one VDF image using the watershed segmentation
    implemented in skimage [1].

    Parameters
    ----------
    VDF_temp : ndarray
        One VDF image.

    min_distance: int
        Minimum distance (in pixels) between grains in order to consider them as
        separate.

    threshold : float
        Threhsold value between 0-1 for the VDF image. Pixels with values below
        (threshold*max intensity in VDF) are discarded and not considered in the
        separation.

    discard_size : float
        Grains with size (total number of pixels) below discard_size are
        discarded.

    plot_on : bool
        If Ture, the VDF, the thresholded VDF, the distance transform and the
        separated grains will be plotted in one figure window.

    Returns
    -------
    sep : ndarray
        Array containing boolean images of separated grains.
        Shape: (image size, image size, number of grains)

    References
    ----------
    [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

    """
    mask = VDF_temp > (threshold * np.max(VDF_temp))

    distance = distance_transform_edt(mask)

    local_maxi = peak_local_max(distance,
                                indices=False,
                                min_distance=min_distance,
                                #threshold_rel=threshold,
                                exclude_border=0,
                                labels=mask)

    labels = watershed(-distance,
                       markers=label(local_maxi)[0],
                       mask=mask)

    if not np.max(labels):
        print('No objects were found. Check parameters.')

    if plot_on:
        axes = hs.plot.plot_images([hs.signals.Signal2D(VDF_temp),
                                    hs.signals.Signal2D(mask),
                                    hs.signals.Signal2D(distance),
                                    hs.signals.Signal2D(labels)],
                                    axes_decor='off',
                                    per_row=2,
                                    colorbar=True,
                                    cmap='nipy_spectral',
                                    label=['VDF0', 'Mask',
                                            'Distances', 'Separated particles'])

    sep=np.empty((np.shape(VDF_temp)[1],np.shape(VDF_temp)[0],(np.max(labels))),dtype=bool)

    n=1
    i=0
    while (np.max(labels)) > n-1:

        sep_temp=labels*(labels==n)/(n)
        sep_temp=np.nan_to_num(sep_temp)
        sep[:,:,(n-i)-1]=(sep_temp.T)

        if np.sum(sep_temp,axis=(0,1)) <= discard_size:
            sep = np.delete(sep,((n-i)-1),axis=2)
            i=i+1

        n=n+1

    return sep

# TODO: This class really needs to keep the g-vector with the corresponding VDF
# image.
class VDFImage(Signal2D):
    _signal_type = "vdf_stack"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def separate_stack(self,
                       min_distance = 3,
                       threshold = 0.4,
                       discard_size = 3,
                       plot_on = False):
        """Separate grains from a stack of images using the watershed
        segmentation implemented in skimage [1], by mapping the function
        separate onto the stack.

        Parameters
        ----------
        min_distance: int
            Minimum distance (in pixels) between features, e.g. grains, in order
            to consider them as separate.

        threshold : float
            Threhsold value between 0-1 for each image. Pixels with values below
            (threshold*max intensity in the image) are discarded and not
            considered in the separation.

        discard_size : float
            Grains (features) with length below discard_size are discarded.
        plot_on : bool
            If Ture, the image, the thresholded image, the distance transform
            and the separated grains (features) will be plotted in ONE FIGURE
            WINDOW FOR EACH IMAGE IN THE STACK.

        Returns
        -------
        seps_stack : ndarray
            Stack of boolean images of separated grains from each VDF image.

        References
        ----------
        [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

        """
        return self.map(separate,
                        show_progressbar=True,
                        parallel=None,
                        inplace=False,
                        ragged=None,
                        min_distance=min_distance,
                        threshold=threshold,
                        discard_size=discard_size,
                        plot_on=plot_on),
                        dtype=np.object)

    def merge_seps_gvector(self,
                           seps_temp,
                           g_temp):
        """Merges separated grain images with its corresponding g-vector.

        Parameters
        ----------
        seps_temp: ndarray
            Stack of separated grains, made from applying the function separate
            on one image.
        g_temp : array
            Unique g-vector [x,y] corresponding to seps_temp.

        Returns
        -------
        merge : ndarray
            An array with two columns, where the first holds the images of the
            separated grains, and the second holds the corresponding g-vector.

        References
        ----------
        [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

        """
        merge=np.empty((np.shape(seps_temp)[2],2), dtype=np.object)
        for i in range(np.shape(seps_temp)[2]):
            merge[i,0]=seps_temp[:,:,i]
            merge[i,1]=g_temp[:]
        return merge

    def merge_seps_gvector_stacks(self, seps_stack_temp,g_vector_stack_temp):
        """Makes a ndarray object holding all the separated grains images and
        corresponding g-vectors.

        Parameters
        ----------
        seps_stack_temp: ndarray
            Stack of separated grains, made from applying the function
            separate_stack on a stack of VDF images.
        g_vector_stack_temp : ndarray
            Array of unique vectors that corresponds to the VDF images that
            seps_temp originates from.

        Returns
        -------
        merge_stack : ndarray
            An array object with two columns, where the first holds all the
            separated grains, and the second the corresponding g-vector for
            each. Shape: (numer of images, 2)
        """
        merge_stack=merge_seps_gvector(seps_stack_temp[0],
                                       g_vector_stack_temp[0])

        for i in range(1,np.shape(seps_stack_temp)[0]):
            merge_stack=np.append(merge_stack,
                                  merge_seps_gvector(seps_stack_temp[i],g_vector_stack_temp[i]), axis=0)

        return merge_stack

    def get_image_stack_from_merge_stack(merge_stack):
        """Reveals the stack of images from a stack consisting of images of the
        same shape in the first column, e.g. a merge_stack with images of
        separated grains.

        Parameters
        ----------
        merge_stack_temp: ndarray
            An array object where each cell in its first column contains images
            of the same shape.

        Returns
        -------
        image_stack : ndarray
            Stack of images with shape (image length in px, image length in px,
            number of images).
        """
        merge_stack=merge_stack[:,0].copy()
        im_shape=np.shape(merge_stack[0])
        num_of_im=np.shape(merge_stack)[0]
        image_stack=np.empty((np.append(im_shape,num_of_im)),dtype=bool)
        for i in range(num_of_im):
            image_stack[:,:,i]=merge_stack[i]
        return image_stack
