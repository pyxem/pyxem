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

"""This contains the data and registry for the data shipped with pyXem.
This data can be used to test pyXem and to provide examples.

"""


_zenodo_url = "https://zenodo.org/records/14113591/files"
# file name : hash
_file_names_hash = {
    "au_xgrating_100kX.hspy": "md5:b0733af9d0a272fc1a47e2f56a324fe5",
    "data_processed.zspy": "md5:bfd9f7e524a65c2988e927f6422dedeb",
    "GaAs_mp-2534_conventional_standard.cif": "md5:9abcda3883bafe29a0790994782ad136",
    "GaAs_mp-8883_conventional_standard.cif": "md5:0116734c35e2a221d08e5e49194e3f3f",
    "MgO.cif": "md5:18224100df2e2b72ad9127bc9f201228",
    "mgo_nano.zspy": "md5:38a03c23cec147a9cffab9ea349cd15a",
    "PdNiP.zspy": "md5:b16375d23eda3da60ea2ff36cc11c5bd",
    "sample_with_g.zspy": "md5:65c0a17387a10ca21ebf1812bab67568",
    "twinned_nanowire.hdf5": "md5:2765fa855db8bc011252dbb425facc81",
    "ZrNbPercipitate.zspy": "md5:7012a2bdf564f4fb25299af05412723f",
    "cuzipProcessed.zspy": "md5:829ecb8f765acb6e9a22092339c6a268",
    "colorwheel.txt": "md5:1555136e42cae858be0716f007bda4e4",
    "SPED-Ag.zspy": "md5:8556346543fc19f0ef9bdc0f4a6619b5",
    "PdCuSiCrystalization-zip.zspy": "md5:80ec7f95ec250106c586debf5d814325",
    "FeAl_stripes.zspy": "md5:702cb0c8ff75062c0cb23b3722c2f859",
    "au_xgrating_20cm.tif": "md5:06192653b9f7841f16a29d3d04e0fd06",
    "au.cif": "md5:4cb2856e8ed9ffac34f5fa22424cd2a2",
    "smallPtychography.hspy": "md5:df9376d5c020a23f0f7f51cfe79f303f",
}
# add to _urls and to _file_names_hash
# if you want to download the data from a different location
_urls = {key: _zenodo_url + "/" + key for key in _file_names_hash.keys()}
