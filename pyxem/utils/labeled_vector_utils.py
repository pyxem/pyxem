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

from scipy.spatial.distance import cdist
import numpy as np

def get_vector_dist(v1, v2):
    """Return the average minimum distnace between the list of vectors v1 and v2.
    This is a modified distance matrix which is a measure of the similarity between the two point clouds.
    """
    d = cdist(v1, v2)
    distance = np.mean([np.mean(np.min(d, axis=0)),
                        np.mean(np.min(d, axis=1))])
    return distance


def vectors2image(vectors,
                  image_size,
                  scales,
                  offsets,
                  indexes=[0, 1],
                  ):
    red_points = vectors[:, indexes]
    red_points = np.round((red_points/scales)-offsets)
    red_points = red_points.astype(int)
    im = np.zeros(image_size)
    im[red_points[:, 0], red_points[:, 1]] = 1
    return im


def column_mean(vectors, columns=[0, 1]):
    return np.mean(vectors[:, columns], axis=0)
