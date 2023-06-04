# -*- coding: utf-8 -*-
# Copyright 2016-2023 The pyXem developers
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

from scipy.ndimage import gaussian_filter
def difference_of_gaussians(image,
                            sigma1,
                            sigma2,
                            order=0,
                            mode="reflect",
                            cval=0.0,
                            truncate=4.0
                            ):

    filtered1 = gaussian_filter(image,
                                sigma=sigma1,
                                order=order,
                                mode=mode,
                                cval=cval,
                                truncate=truncate)
    filtered2 = gaussian_filter(image,
                                sigma=sigma2,
                                order=order,
                                mode=mode,
                                cval=cval,
                                truncate=truncate)

    return filtered1-filtered2


def difference_of_gaussians_lazy(image,
                                 sigma1,
                                 sigma2,
                                 order=0,
                                 mode="reflect",
                                 cval=0.0,
                                 truncate=4.0,
                                 **kwargs,
                                 ):
    from dask_image.ndfilters._gaussian import  _get_border, _get_sigmas
    from dask_image.ndfilters._utils import _get_depth_boundary
    sigma2 = _get_sigmas(image, sigma2)
    sigma1 = _get_sigmas(image, sigma1)
    depth = _get_border(image, sigma2, truncate)
    depth, boundary = _get_depth_boundary(image.ndim, depth, "none")
    result = image.map_overlap(difference_of_gaussians,
                               depth=depth,
                               boundary=boundary,
                               dtype=image.dtype,
                               meta=image._meta,
                               sigma1=sigma1,
                               sigma2=sigma2,
                               order=order,
                               mode=mode,
                               cval=cval,
                               truncate=truncate,
                               **kwargs
                               )
    return result