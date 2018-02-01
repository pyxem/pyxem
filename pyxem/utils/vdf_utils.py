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

import numpy as np
from tqdm import tqdm

from scipy.ndimage import distance_transform_edt, label

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.feature import match_template


def separate(VDF_temp,
             min_distance,
             threshold,
             min_size,
             max_size,
             max_number_of_grains,
             exclude_border=0,
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
    min_size : float
        Grains with size (total number of pixels) below min_size are discarded.
    max_size : float
        Grains with size (total number of pixels) above max_size are discarded.
    max_number_of_grains : int
        Maximum number of grains included in separated particles.
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

def norm_cross_corr(image, template):
    """Calculates the normalised cross-correlation between an image and a
    template.

    Parameters
    ----------
    image: ndarray
        A 2D array object.
    template: ndarray
        Another 2D array object.

    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template at zero
        displacement.
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
        All gvectors that should be found at position i in the image correlated
        merge stack.
    add_indices: ndarray
        Indices in the merge stack corresponding to the vectors in
        gvectors_of_add_indices.
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
    """Calculates the normalised cross-correlation between an image and a
    template.

    Parameters
    ----------
    image: ndarray
        A 2D array object.
    template: ndarray
        Another 2D array object.

    Returns
    -------
    corr : float
        Normalised cross-correlation between image and template at zero
        displacement.
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
        All gvectors that should be found at position i in the image correlated
        merge stack.
    add_indices: ndarray
        Indices in the merge stack corresponding to the vectors in
        gvectors_of_add_indices.
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
