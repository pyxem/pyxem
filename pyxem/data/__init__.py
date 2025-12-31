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
import importlib


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
    "simulated_strain",
    "simulated_pn_junction",
    "zrcual_1",
    "zrcual_2",
    "zrcual_3",
    "dummy_data",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "au_grating": "._data",
    "pdnip_glass": "._data",
    "zrnb_precipitate": "._data",
    "twinned_nanowire": "._data",
    "sample_with_g": "._data",
    "mgo_nanocrystals": "._data",
    "tilt_boundary_data": "._simulated_tilt",
    "si_phase": "._simulated_si",
    "si_tilt": "._simulated_si",
    "si_grains": "._simulated_si",
    "si_grains_simple": "._simulated_si",
    "si_rotations_line": "._simulated_si",
    "simulated_stripes": "._simulated_dpc",
    "fe_multi_phase_grains": "._simulated_fe",
    "fe_fcc_phase": "._simulated_fe",
    "fe_bcc_phase": "._simulated_fe",
    "cuag_orientations": "._data",
    "organic_semiconductor": "._data",
    "feal_stripes": "._data",
    "sped_ag": "._data",
    "pdcusi_insitu": "._data",
    "simulated_constant_shift_magnitude": "._simulated_dpc",
    "au_grating_20cm": "._data",
    "small_ptychography": "._data",
    "au_phase": "._data",
    "simulated_overlap": "._simulated_overlap",
    "simulated_strain": "._simulated_strain",
    "simulated_pn_junction": "._simulated_dpc",
    "zrcual_1": "._data",
    "zrcual_2": "._data",
    "zrcual_3": "._data",
    "dummy_data": "",
}


def __getattr__(name):
    if name == "dummy_data":
        return importlib.import_module("pyxem.data.dummy_data")
    if name in __all__:
        import_path = "pyxem.data" + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
