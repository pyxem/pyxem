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

from ._data import (
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
    zrcual_1,
    zrcual_2,
    zrcual_3,
)
from ._simulated_dpc import (
    simulated_stripes,
    simulated_constant_shift_magnitude,
    simulated_pn_junction,
)
from ._simulated_fe import fe_bcc_phase, fe_fcc_phase, fe_multi_phase_grains
from ._simulated_overlap import simulated_overlap
from ._simulated_si import (
    si_phase,
    si_tilt,
    si_grains,
    si_grains_simple,
    si_rotations_line,
)
from ._simulated_strain import simulated_strain
from ._simulated_tilt import tilt_boundary_data
from .dummy_data import dummy_data

__all__ = [
    "au_grating",
    "pdnip_glass",
    "zrnb_precipitate",
    "twinned_nanowire",
    "sample_with_g",
    "mgo_nanocrystals",
    "organic_semiconductor",
    "cuag_orientations",
    "feal_stripes",
    "sped_ag",
    "pdcusi_insitu",
    "au_grating_20cm",
    "small_ptychography",
    "au_phase",
    "zrcual_1",
    "zrcual_2",
    "zrcual_3",
    "simulated_stripes",
    "simulated_constant_shift_magnitude",
    "simulated_pn_junction",
    "fe_bcc_phase",
    "fe_fcc_phase",
    "fe_multi_phase_grains",
    "simulated_overlap",
    "si_phase",
    "si_tilt",
    "si_grains",
    "si_grains_simple",
    "si_rotations_line",
    "simulated_strain",
    "tilt_boundary_data",
    "dummy_data",
]

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
