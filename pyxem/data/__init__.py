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

"""Example datasets for use when testing functionality.

Some datasets are packaged with the source code while others must be downloaded from the web.
For more test datasets, see Open datasets.

Datasets are placed in a local cache, in the location returned from pooch.os_cache("pyxem") by default.
the location can be overwritten with a global PYXEM_DATA_DIR environment variable.

With every new version of pyxem, a new directory of datasets with the version name
is added to the cache directory. Any old directories are not deleted automatically, and should
then be deleted manually if desired.
"""

from pyxem.data.simulated_tilt import tilt_boundary_data
from pyxem.data.simulated_si import (
    si_phase,
    si_tilt,
    si_grains,
    si_grains_simple,
    si_rotations_line,
)

from pyxem.data.simulated_overlap import simulated_overlap

from pyxem.data.simulated_fe import fe_bcc_phase, fe_fcc_phase, fe_multi_phase_grains
from pyxem.data._data import (
    au_grating,
    pdnip_glass,
    zrnb_precipitate,
    twinned_nanowire,
    sample_with_g,
    mgo_nanocrystals,
    organic_semiconductor,
    cuag_orientations,
    feal_stripes,
    sped_ag,
    pdcusi_insitu,
    au_grating_20cm,
    small_ptychography,
    au_phase,
)
from pyxem.data.simulated_dpc import (
    simulated_stripes,
    simulated_constant_shift_magnitude,
)

__all__ = [
    "au_grating",
    "pdnip_glass",
    "zrnb_precipitate",
    "twinned_nanowire",
    "sample_with_g",
    "mgo_nanocrystals",
    "tilt_boundary_data",
    "si_phase",
    "si_tilt",
    "si_grains",
    "si_grains_simple",
    "si_rotations_line",
    "simulated_stripes",
    "fe_multi_phase_grains",
    "fe_fcc_phase",
    "fe_bcc_phase",
    "cuag_orientations",
    "organic_semiconductor",
    "feal_stripes",
    "sped_ag",
    "pdcusi_insitu",
    "simulated_constant_shift_magnitude",
    "au_grating_20cm",
    "small_ptychography",
    "au_phase",
    "simulated_overlap",
]
