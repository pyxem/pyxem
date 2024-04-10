# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

import warnings

warnings.warn(
    "This module has been renamed and should now be imported as `pyxem.utils.diffraction`",
    FutureWarning,
)

from pyxem.utils.diffraction import *


def match_template_dilate(
    image, template, template_dilation=2, mode="constant", constant_values=0
):
    """
    Matches a template with an image using a window normalized cross-correlation. This preforms very well
    for image with different background intensities.  This is a slower version of the skimage `match_template`
    but performs better for images with circular variations in background intensity, specifically accounting
    for an amorphous halo around the diffraction pattern.

    Parameters
    ----------
    image : np.array
        Image to be matched
    template : np.array
        Template to preform the normalized cross-correlation with
    template_dilation : int
        The number of pixels to dilate the template by for the windowed cross-correlation
    mode : str
        Padding mode for the image. Options are 'constant', 'edge', 'wrap', 'reflect'
    constant_values : int
        Value to pad the image with if mode is 'constant'
    """

    if image.ndim < template.ndim:
        raise ValueError(
            "Dimensionality of template must be less than or "
            "equal to the dimensionality of image."
        )
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape
    float_dtype = _supported_float_type(image.dtype)

    template = np.pad(template, template_dilation)
    pad_width = tuple((width, width) for width in template.shape)
    if mode == "constant":
        image = np.pad(
            image, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    else:
        image = np.pad(image, pad_width=pad_width, mode=mode)

    dilated_template = morphology.dilation(
        template, footprint=morphology.disk(template_dilation)
    )
    # Use special case for 2-D images for much better performance in
    # computation of integral images
    image_window_sum = fftconvolve(image, dilated_template[::-1, ::-1], mode="valid")[
        1:-1, 1:-1
    ]
    image_window_sum2 = fftconvolve(
        image**2, dilated_template[::-1, ::-1], mode="valid"
    )[1:-1, 1:-1]

    template_mean = template.mean()
    template_volume = np.sum(dilated_template)
    template_ssd = np.sum((template - template_mean) ** 2)

    xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]
    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=float_dtype)

    # avoid zero-division
    mask = denominator > np.finfo(float_dtype).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + image_shape[i]

        slices.append(slice(d0, d1))

    return response[tuple(slices)]
