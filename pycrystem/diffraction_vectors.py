# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

from hyperspy.api import roi
from hyperspy.signals import BaseSignal, Signal1D, Signal2D

from .utils.expt_utils import *

from tqdm import tqdm

from scipy.ndimage import distance_transform_edt, label

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.feature import match_template
"""
Signal class for diffraction vectors.
"""

def _calculate_norms(z):
    norms = []
    #print(z)
    for i in z[0]:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)

def get_new_unique_vectors_in_list(vlist,
                               gvlist,
                              distance_threshold=0):
    """Obtain a list of unique diffraction vectors from vlist, that are not found in gvlist.
    NB This function is used by get_unique_vectors

    Parameters
    ----------
    vlist : ndarray
        List of vectors.
    distance_threshold : float
        The minimum distance between gvectors for them to be considered as
        different gvectors.
    gvlist : ndarray
        List of unique vectors to be compared to the diffraction vectors in vlist.

    Returns
    -------
    unique_vectors : list
        List of unique diffraction vectors from vlist, that are not found in gvlist. 
        None will be returned if there are no such vectors.

    """
    if sum(map(lambda x: np.allclose(vlist,
                                      x, 
                                      rtol=0, 
                                      atol=distance_threshold, 
                                      equal_nan=False), 
               gvlist)):
        pass
    else:
        return np.asarray(vlist)

class DiffractionVectors(BaseSignal):
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)

    def plot(self):
        """Plot the diffraction vectors.
        """
        #Find the unique gvectors to plot.
        unique_vectors = self.get_unique_vectors()
        #Plot the gvector positions
        import matplotlib.pyplot as plt
        plt.plot(unique_vectors.T[1], unique_vectors.T[0], 'ro')
        plt.axes().set_aspect('equal')
        plt.show()

    def get_magnitudes(self):
        """Calculate the magnitude of diffraction vectors.

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containging an array of gvector magnitudes at each
            navigation position.

        """
        magnitudes = self.map(_calculate_norms, inplace=False)
        return magnitudes

    def get_magnitude_histogram(self, bins):
        """Obtain a histogram of gvector magnitudes.

        Parameters
        ----------
        bins : numpy array
            The bins to be used to generate the histogram.

        Returns
        -------
        ghist : Signal1D
            Histogram of gvector magnitudes.

        """
        gnorms = self.get_magnitudes()

        glist=[]
        for i in gnorms._iterate_signal():
            for j in np.arange(len(i[0])):
                glist.append(i[0][j])
        gs = np.asarray(glist)
        gsig = Signal1D(gs)
        ghis = gsig.get_histogram(bins=bins)
        ghis.axes_manager.signal_axes[0].name = 'g-vector magnitude'
        ghis.axes_manager.signal_axes[0].units = '$A^{-1}$'
        return ghis

    def get_unique_vectors(self,
                           distance_threshold=None,
                           x=0,y=0,z=0):
        """Obtain a unique list of diffraction vectors.

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between gvectors for them to be considered as
            different gvectors.
        x,y,z : int
            Integers defining the position of the g-vector to use as the first
            vector in the list of unique vectors. 
            Notation: self.inav[x,y].data[z] or self.inav[x].data[z] 

        Returns
        -------
        unique_vectors : list
            List of all unique diffraction vectors.
        """
        #Pick one gvector defined by x,y,z, as staring point for gvlist. 
        if np.shape(np.shape(self.axes_manager))[0] >= 2:
            gvlist=np.asarray([self.inav[x,y].data[z]])
        else:
            gvlist=np.asarray([self.inav[x].data[z]])

        #Iterate through self, find and append all unique vectors to gvlist.
        for i in tqdm(self._iterate_signal()):
            gvlist_new=list(map(lambda x: get_new_unique_vectors_in_list(x,
                                                                     gvlist,
                                                                     distance_threshold=distance_threshold),
                            np.asarray(i[0])))
            #For all vectors in i that are not unique, gvlist_new will include None values. Those are deleted. 
            gvlist_new= list(filter(lambda x: x is not None, gvlist_new))
            gvlist_new=np.reshape(gvlist_new, newshape=(-1,2))

            #If gvlist_new contain new unique vectors, add these to gvlist. 
            if gvlist_new.any():
                gvlist=np.concatenate((gvlist, gvlist_new),axis=0)

        unique_vectors = np.asarray(gvlist)
        return unique_vectors

    def get_vdf_images(self,
                       electron_diffraction,
                       radius,
                       unique_vectors=None):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an ElectronDiffraction Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        unique_vectors : list (optional)
            Unique list of diffracting vectors if pre-calculated. If None the
            unique vectors in self are determined and used.

        electron_diffraction : ElectronDiffraction
            ElectronDiffraction signal from which to extract the reflection
            intensities.

        radius : float
            Radius of the integration window summed over in reciprocal angstroms.

        Returns
        -------
        vdfs : Signal2D
            Signal containing virtual dark field images for all unique g-vectors.
        """
        if unique_vectors==None:
            unique_vectors = self.get_unique_vectors()
        else:
            unique_vectors = unique_vectors

        vdfs = []
        for v in unique_vectors:
            disk = roi.CircleROI(cx=v[1], cy=v[0], r=radius, r_inner=0)
            vdf = disk(electron_diffraction,
                       axes=electron_diffraction.axes_manager.signal_axes)
            vdfs.append(vdf.sum((2,3)).as_signal2D((0,1)).data)
        VDFs = Signal2D(np.asarray(vdfs))
        return VDFStack(VDFs)

    def get_gvector_indexation(self,
                               calculated_peaks,
                               magnitude_threshold,
                               angular_threshold=None):
        """Index diffraction vectors based on the magnitude of individual
        vectors and optionally the angles between pairs of vectors.

        Parameters
        ----------

        calculated_peaks : array
            Structured array containing the theoretical diffraction vector
            magnitudes and angles between vectors.

        magnitude_threshold : Float
            Maximum deviation in diffraction vector magnitude from the
            theoretical value for an indexation to be considered possible.

        angular_threshold : float
            Maximum deviation in the measured angle between vector
        Returns
        -------

        gindex : array
            Structured array containing possible indexations
            consistent with the data.

        """
        #TODO: Specify threshold as a fraction of the g-vector magnitude.
        arr_shape = (self.axes_manager._navigation_shape_in_array
                     if self.axes_manager.navigation_size > 0
                     else [1, ])
        gindex = np.zeros(arr_shape, dtype=object)

        for i in self.axes_manager:
            it = (i[1], i[0])
            res = []
            for j in np.arange(len(glengths[it])):
                peak_diff = (calc_peaks.T[1] - glengths[it][j]) * (calc_peaks.T[1] - glengths[it][j])
                res.append((calc_peaks[np.where(peak_diff < magnitude_threshold)],
                            peak_diff[np.where(peak_diff < magnitude_threshold)]))
            gindex[it] = res

        if angular_threshold==None:
            pass
        else:
            pass

        return gindex

    def get_zone_axis_indexation(self):
        """Determine the zone axis consistent with the majority of indexed
        diffraction vectors.

        Parameters
        ----------

        Returns
        -------

        """

def separate(VDF_temp,
             min_distance,
             threshold,
             min_size,
             max_size,
             max_number_of_grains,
             exclude_border=0,
             plot_on = False):
    """Separate grains from one VDF image using the watershed segmentation implemented in skimage [1].

    Parameters
    ----------
    VDF_temp : ndarray
        One VDF image.
    min_distance: int
        Minimum distance (in pixels) between grains in order to consider them as separate. 
    threshold : float
        Threhsold value between 0-1 for the VDF image. Pixels with values below 
        (threshold*max intensity in VDF) are discarded and not considered in the separation. 
    min_size : float
        Grains with size (total number of pixels) below min_size are discarded.
    max_size : float
        Grains with size (total number of pixels) above max_size are discarded.
    max_number_of_grains : int
        Maximum number of grains included in separated particles. 
    plot_on : bool
        If Ture, the VDF, the thresholded VDF, the distance transform and the separated grains
        will be plotted in one figure window. 
        
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
    distance = distance_transform_edt(VDF_temp)
    local_maxi = peak_local_max(distance, 
                                indices=False,
                                min_distance=min_distance,
                                num_peaks = max_number_of_grains,
                                exclude_border=exclude_border,
                                labels=mask)
    labels = watershed(-distance, 
                       markers=label(local_maxi)[0],
                       mask=mask)
    if not np.max(labels):
        print('No labels were found. Check parameters.')
    sep=np.zeros((np.shape(VDF_temp)[1],np.shape(VDF_temp)[0],(np.max(labels))),dtype='int32')
    n=1
    i=0
    while (np.max(labels)) > n-1:
        sep_temp=labels*(labels==n)/(n)
        sep_temp=np.nan_to_num(sep_temp)
        sep[:,:,(n-i)-1]=(sep_temp.T)
        if (np.sum(sep_temp,axis=(0,1)) < min_size) or ((max_size != None) and (np.sum(sep_temp,axis=(0,1)) > max_size)):
            sep = np.delete(sep,((n-i)-1),axis=2)
            i=i+1    
        n=n+1
    VDF_sep = np.reshape(np.tile(VDF_temp,(np.shape(sep)[2])),
                         newshape=(np.shape(sep))) * (sep==1)

    if plot_on:
        #If particles have been discarded, make new labels that does not include these
        if np.max(labels) != (np.shape(sep)[2]) and (np.shape(sep)[2] != 0):
            #if np.shape(sep[0,0])[0] > 1:
            labels = sep[:,:,0]
            for i in range(1,np.shape(sep)[2]):
                labels = labels + sep[...,i]*(i+1)
            labels = labels.T
        #If no separated particles were found, set all elements in labels to 0.
        elif (np.shape(sep)[2] == 0):
            labels = np.zeros(np.shape(labels))
            print('No separate particles were found.')
        axes = hs.plot.plot_images([hs.signals.Signal2D(VDF_temp),
                                    hs.signals.Signal2D(mask), 
                                    hs.signals.Signal2D(distance), 
                                    hs.signals.Signal2D(labels),
                                    hs.signals.Signal2D(np.sum(VDF_sep,axis=2).T)],
                                    axes_decor='off', 
                                    per_row=3, 
                                    colorbar=True, 
                                    cmap='gnuplot2',
                                    label=['VDF', 'Mask', 'Distances', 
                                    'Labels','Separated particles'])    
    return VDF_sep   

class VDFStack(Signal2D):
    _signal_type = "image_stack"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def separate_stack(self,
                       min_distance,
                       threshold,
                       min_size,
                       max_size,
                       max_number_of_grains = np.inf,
                       exclude_border=0):
        """Separate grains from a stack of images using the watershed segmentation implemented in skimage [1], 
        by mapping the function separate onto the stack. 

        Parameters
        ----------
        min_distance: int
            Minimum distance (in pixels) between features, e.g. grains, in order to consider them as separate. 
        threshold : float
            Threhsold value between 0-1 for each image. Pixels with values below 
            (threshold*max intensity in the image) are discarded and not considered in the separation. 
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

class SepsStack(BaseSignal):
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
                Array of unique vectors that corresponds to the VDF images that seps_temp originates from. 
                
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

def norm_cross_corr(image, template):
    """Calculates the normalised cross-correlation between an image and a template. 

    Parameters
    ----------
    image: ndarray
        A 2D array object.   
    template: ndarray
        Another 2D array object.
        
    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template at zero displacement.
    """
    #If image and template are alike, 1 will be returned.
    if np.array_equal(image,template):
        corr=1.
    else: 
        #Return only the value in the middle, i.e. at zero displcement
        corr = match_template(image=image, 
                              template=template, 
                              pad_input=True, 
                              mode='constant', 
                              constant_values=0)[int(np.shape(image)[0]/2),int(np.shape(image)[1]/2)]
    return corr

def corr_check(corr,corr_threshold):
    """Checks if a value is above a threshold. 
    
    Parameters
    ----------
    corr: float
        Value to be checked.
    corr_threshold: float
        Threshold value. 

    Returns
    -------
    add : bool
        True if corr is above corr_threhsold.
    """
    if corr>corr_threshold:
        add=True
    else: 
        add=False
    return add    

def make_g_of_i(gvectors_of_add_indices,add_indices,gvector_i):
    """Makes an array containing the gvectors to be placed at position i
    in an image correlated merge stack.

    Parameters
    ----------
    gvectors_of_add_indices: ndarray
        All gvectors that should be found at position i in the image correlated merge stack.
    add_indices: ndarray
        Indices in the merge stack corresponding to the vectors in gvectors_of_add_indices. 
    gvector_i: ndarray
        Array of gvectors formerly at position i. 
        
    Returns
    -------
    g : ndarray
        Gvectors to be placed at position i in the image correlated merge stack.
    """
    if len(np.shape(gvector_i))==1:
        g=np.array([gvector_i])
    else: 
        g=gvector_i
    
    for i in range(np.shape(add_indices)[0]):
        if len(np.shape(gvectors_of_add_indices[i]))==1:
            g=np.append(g,np.array([gvectors_of_add_indices[i]]),axis=0)
            
        elif len(np.shape(gvectors_of_add_indices[i])) == 2:
            g=np.append(g,gvectors_of_add_indices[i],axis=0)

        elif len(np.shape(gvectors_of_add_indices[i])) == 3:
            for n in range(np.shape(gvectors_of_add_indices[i][0])[0]):
                g=np.append(g,gvectors_of_add_indices[i][n])
            
    g_delete=[]
    
    for i in range(np.shape(g)[0]):
        
        g_in_list = sum(map(lambda x: np.array_equal(g[i],x), 
                           g[i+1:]))
        if g_in_list:
            g_delete = np.append(g_delete,i)
    g = np.delete(g, g_delete,axis=0)
    return g

def norm_cross_corr(image, template):
    """Calculates the normalised cross-correlation between an image and a template. 

    Parameters
    ----------
    image: ndarray
        A 2D array object.   
    template: ndarray
        Another 2D array object.
        
    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template at zero displacement.
    """
    #If image and template are alike, 1 will be returned.
    if np.array_equal(image,template):
        corr=1.
    else: 
        #Return only the value in the middle, i.e. at zero displcement
        corr = match_template(image=image, 
                              template=template, 
                              pad_input=True, 
                              mode='constant', 
                              constant_values=0)[int(np.shape(image)[0]/2),int(np.shape(image)[1]/2)]
    return corr

def corr_check(corr,corr_threshold):
    """Checks if a value is above a threshold. 
    
    Parameters
    ----------
    corr: float
        Value to be checked.
    corr_threshold: float
        Threshold value. 

    Returns
    -------
    add : bool
        True if corr is above corr_threhsold.
    """
    if corr>corr_threshold:
        add=True
    else: 
        add=False
    return add   

def make_g_of_i(gvectors_of_add_indices,add_indices,gvector_i):
    """Makes an array containing the gvectors to be placed at position i
    in an image correlated merge stack.

    Parameters
    ----------
    gvectors_of_add_indices: ndarray
        All gvectors that should be found at position i in the image correlated merge stack.
    add_indices: ndarray
        Indices in the merge stack corresponding to the vectors in gvectors_of_add_indices. 
    gvector_i: ndarray
        Array of gvectors formerly at position i. 
        
    Returns
    -------
    g : ndarray
        Gvectors to be placed at position i in the image correlated merge stack.
    """
    if len(np.shape(gvector_i))==1:
        g=np.array([gvector_i])
    else: 
        g=gvector_i
    
    for i in range(np.shape(add_indices)[0]):
        if len(np.shape(gvectors_of_add_indices[i]))==1:
            g=np.append(g,np.array([gvectors_of_add_indices[i]]),axis=0)
            
        elif len(np.shape(gvectors_of_add_indices[i])) == 2:
            g=np.append(g,gvectors_of_add_indices[i],axis=0)

        elif len(np.shape(gvectors_of_add_indices[i])) == 3:
            for n in range(np.shape(gvectors_of_add_indices[i][0])[0]):
                g=np.append(g,gvectors_of_add_indices[i][n])
            
    g_delete=[]
    
    for i in range(np.shape(g)[0]):
        
        g_in_list = sum(map(lambda x: np.array_equal(g[i],x), 
                           g[i+1:]))
        if g_in_list:
            g_delete = np.append(g_delete,i)
    g = np.delete(g, g_delete,axis=0)
    return g

class VDFgvectorStack():
    '''Class for which VDFgvectorStack.images holds all the VDF images of the separated grains,
    and VDFgvectorStack.vectors the corresponding g-vector for each image.'''
    _signal_type = "image_vector_stack"

    def __init__(self, images, vectors, *args,**kwargs):
        self.images = Signal2D(images)
        self.vectors = DiffractionVectors(vectors)
        self.vectors.axes_manager.set_signal_dimension(0)

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

    def manage_images_and_gvectors_at_indices(self,
                                              image_add_indices = None,
                                              gvectors_add_indices = None,
                                              delete_indices = None):
        """Sums or deletes images, or adds gvectors, with the given indices, 
            from a merge stack (stack of images and corresponding gvectors that are found in the first 
            and second column respectively). 

        Parameters
        ----------
        image_add_indices: int
            Indices for the images to be summed. Corresponding gvectors will also be added. 
            Example: To sum the images at 1 and 2: [[1,2]]. To sum the images at 1 and 2, and 
            those at 5 and 6: [[1,2],[5,6]]
        gvectors_add_indices: int
            Indices for the gvectors that will be added. Corresponding images will not be added.
        delete_indices: int
            Indices for the images and gvectors to be deleted. Example: To delete 2 and 3: [2,3]
            
        Returns
        -------
        VDFgvectorStack
            The VDFgvectorStack class instance updated according to the addition and/or deletion.
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
        