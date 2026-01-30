# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

"""Example datasets for use when testing functionality.

Some datasets are packaged with the source code while others must be downloaded from the web.
For more test datasets, see Open datasets.

Datasets are placed in a local cache, in the location returned from pooch.os_cache("pyxem") by default.
the location can be overwritten with a global PYXEM_DATA_DIR environment variable.

With every new version of pyxem, a new directory of datasets with the version name
is added to the cache directory. Any old directories are not deleted automatically, and should
then be deleted manually if desired.
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__doctest_skip__ = [
    "au_grating",
    "au_grating_20cm",
    "cuag_orientations",
    "pdnip_glass",
    "zrnb_precipitate",
    "twinned_nanowire",
    "sample_with_g",
    "organic_semiconductor",
    "pdcusi_insitu",
    "feal_stripes",
    "sped_ag",
    "mgo_nanocrystals",
    "zrcual_1",
    "zrcual_2",
    "zrcual_3",
    "small_ptychography",
]
