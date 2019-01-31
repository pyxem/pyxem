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
from hyperspy.signals import BaseSignal, Signal2D
from pyxem.utils.vdf_utils import (normalize_vdf, norm_cross_corr, corr_check,
                                   make_g_of_i)
from pyxem.signals.diffraction_vectors import DiffractionVectors

class VDFImage(Signal2D):
    _signal_type = "vdf_image"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vectors = None


class VDFSegment:
    '''Class for which VDFgvectorStack.images holds all the VDF images of the separated grains,
    and VDFgvectorStack.vectors the corresponding g-vector for each image.'''
    _signal_type = "image_vector_stack"

    def __init__(self, segments, vectors_of_segments, *args,**kwargs):
        self.segments = segments
        self.vectors_of_segments = vectors_of_segments
        self.vectors_of_segments.axes_manager.set_signal_dimension(0)

    def image_correlate_stack(self,corr_threshold=0.9):
        """Iterates through VDFgvectorStack, and sums those that are associated with the same grains. 
            Summation will be done for those images that has a normalised cross correlation above the threshold. 
            The gvectors of each grain will be updated accordingly. 
        Parameters
        ----------
        corr_threshold: float
            Threshold value for the image cross correlation value for images to be added together, 
            e.g. to be considered the same grain. 
        Returns
        -------
        VDFgvectorStack
            The VDFgvectorStack class instance updated according to the image correlation results.
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
