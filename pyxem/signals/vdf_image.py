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

import numpy as np
from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from pyxem.utils.vdf_utils import *


class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def separate_stack(self,
                       min_distance,
                       threshold,
                       min_size,
                       max_size,
                       max_number_of_grains=1000,
                       exclude_border=0):
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

        Returns
        -------
        seps_stack : ndarray
            Stack of boolean images of separated grains from each VDF image.

        References
        ----------
        [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

        TODO: Enable plot of separate stack as in separate!
        """
        return SepsStack(np.array((self).map(separate,
                                            show_progressbar=True,
                                            parallel=None,
                                            inplace=False,
                                            ragged=None,
                                            min_distance=min_distance,
                                            threshold=threshold,
                                            min_size=min_size,
                                            max_size=max_size,
                                            max_number_of_grains=max_number_of_grains,
                                            exclude_border=exclude_border,
                                            plot_on=False),
                                            dtype=np.object))

class VDFSegmentation(BaseSignal):
    """Stack of separated grains, made from applying the function separate_stack
       on a stack of VDF images. """
    _signal_type = "separated_image_stack"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)

    def get_VDFgvectorStack(self,unique_vectors):

        """Makes a image_vector_stack class instance,
         holding all the separated grains images and corresponding g-vectors.

        Parameters
        ----------
        unique_vectors : ndarray
            Array of unique vectors that corresponds to the VDF images that
            seps_temp originates from.

        Returns
        -------
        image_vector_stack
        """
        images = self.data[0].T
        vectors=np.array(np.empty((1)),dtype='object')
        vectors[0]=unique_vectors[0]
        for i in range(1,np.shape(self)[0]):
            images = np.append(images, self.data[i].T,axis = 0)
            repeats=np.array(np.empty((np.shape(self.data[i])[2])),dtype='object')
            for n in range(np.shape(self.data[i])[2]):
                repeats[n] = unique_vectors[i]
            vectors = np.append(vectors,repeats,axis=0)
        return VDFgvectorStack(images,vectors)


class VDFgvectorStack():
    '''Class for which VDFgvectorStack.images holds all the VDF images of the
    separated grains, and VDFgvectorStack.vectors the corresponding g-vector for
    each image.'''
    _signal_type = "image_vector_stack"

    def __init__(self, images, vectors, *args,**kwargs):
        self.images = Signal2D(images)
        #TODO:This was DiffractionVectors but now circular import - check
        #implications
        self.vectors = BaseSignal(vectors)
        self.vectors.axes_manager.set_signal_dimension(0)

    def image_correlate_stack(self,corr_threshold=0.9):
        """Iterates through VDFgvectorStack, and sums those that are associated
        with the same grains. Summation will be done for those images that has a
        normalised cross correlation above the threshold. The gvectors of each
        grain will be updated accordingly.

        Parameters
        ----------
        corr_threshold: float
            Threshold value for the image cross correlation value for images to
            be added together, e.g. to be considered the same grain.

        Returns
        -------
        VDFgvectorStack
            The VDFgvectorStack class instance updated according to the image
            correlation results.
        """
        image_stack=self.images.data
        gvectors=self.vectors.data

        i=0
        pbar = tqdm(total=np.shape(image_stack)[0])
        while np.shape(image_stack)[0]>i:
            corr_list=list(map(lambda x: norm_cross_corr(x, template=image_stack[i]), image_stack))
            corr_add=list(map(lambda x: corr_check(x,corr_threshold=corr_threshold), corr_list))
            add_indices=np.where(corr_add)

            if np.shape(add_indices[0])[0] > 1:
                image_stack[i]=np.sum(list(map(lambda x: np.sum([x,image_stack[i]],axis=0),
                                               image_stack[add_indices])),
                                                axis=0)

                add_indices=add_indices[0]

                gvectors[i] = make_g_of_i(gvectors[add_indices],add_indices,gvectors[i])

                add_indices_noi=np.delete(add_indices,np.where(add_indices==i),axis=0)
                image_stack=np.delete(image_stack, add_indices_noi, axis=0)
                gvectors=np.delete(gvectors, add_indices_noi, axis=0)
            else:
                add_indices_noi=add_indices

            if np.where(add_indices == i) != np.array([0]):
                i = i+1 - (np.shape(np.where(add_indices < i))[1])
            else:
                i=i+1

            if len(np.shape(gvectors[i-1])) == 1:
                gvectors[i-1]=np.array([gvectors[i-1]])
            pbar.update(np.shape(add_indices_noi)[0])
        pbar.close()
        return VDFgvectorStack(image_stack,gvectors)

    def get_virtual_electron_diffraction_signal(self,
                                                electron_diffraction,
                                                distance_threshold=None,
                                                A = 255):
        """ Created an ElectronDiffraction signal consisting of Gaussians at all
        the gvectors.

        Parameters
        ----------
        electron_diffraction: ElectronDiffraction
            ElectronDiffraction signal that the merge_stack_corr originates from.
        distance_threshold : float
            The FWHM in the 2D Gaussians that will be calculated for each
            g-vector, in order to create the virtual DPs. It is reasonable to
            choose this value equal to the distance_threshold that was used to
            find the unique g-vectors, and thus the name.

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

    def manage_images_and_gvectors_at_indices(self,
                                              image_add_indices = None,
                                              gvectors_add_indices = None,
                                              delete_indices = None):
        """Sums or deletes images, or adds gvectors, with the given indices,
            from a merge stack (stack of images and corresponding gvectors that
            are found in the first and second column respectively).

        Parameters
        ----------
        image_add_indices: int
            Indices for the images to be summed. Corresponding gvectors will
            also be added.
            Example: To sum the images at 1 and 2: [[1,2]]. To sum the images at
            1 and 2, and those at 5 and 6: [[1,2],[5,6]]
        gvectors_add_indices: int
            Indices for the gvectors that will be added. Corresponding images
            will not be added.
        delete_indices: int
            Indices for the images and gvectors to be deleted. Example: To
            delete 2 and 3: [2,3]

        Returns
        -------
        VDFgvectorStack
            The VDFgvectorStack class instance updated according to the addition
            and/or deletion.
        """
        image_stack=self.images.data.copy()
        gvectors=self.vectors.data.copy()

        if np.any(image_add_indices):
            for i in image_add_indices:
                image_stack[i[0]]=np.sum(list(map(lambda x: np.sum([x, image_stack[i[0]]], axis=0),
                                            image_stack[i[1:]])),
                                            axis=0)
                gvectors[i[0]] = make_g_of_i(gvectors[i], i, gvectors[i[0]])

        if np.any(gvectors_add_indices):
            for i in gvectors_add_indices:
                gvectors[i[0]] = make_g_of_i(gvectors[i], i, gvectors[i[0]])

        if delete_indices is not None:
            image_stack=np.delete(image_stack, delete_indices, axis=0)
            gvectors=np.delete(gvectors, delete_indices, axis=0)
            if not np.shape(image_stack)[0]:
                print('No stack left after deletion. Check delete_indices.')

        if not np.any(delete_indices) and not np.any(gvectors_add_indices) and not np.any(image_add_indices):
            print('Specify indices for addition or deletion.')

        return VDFgvectorStack(image_stack,gvectors)

    def threshold_VDFgvectorStack(self,
                                  image_threshold=None,
                                  gvector_threshold=None):
        image_stack = self.images.data.copy()
        gvectors = self.vectors.data.copy()

        if image_threshold is not None:
            n=0
            while np.shape(image_stack)[0] > n:
                if np.max(image_stack[n]) < image_threshold:
                    image_stack = np.delete(image_stack,n,axis=0)
                    gvectors = np.delete(gvectors,n,axis=0)
                else:
                    n=n+1
        if gvector_threshold is not None:
            n=0
            while np.shape(image_stack)[0] > n:
                if np.shape(gvectors[n])[0] < gvector_threshold:
                    image_stack = np.delete(image_stack,n,axis=0)
                    gvectors = np.delete(gvectors,n,axis=0)
                else:
                    n=n+1
        if not np.any(image_stack):
            print('No stack left after thresholding. Check thresholds.')
            return 0
        return VDFgvectorStack(image_stack,gvectors)
