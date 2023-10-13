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


from pyxem.data import (
    pdnip_glass,
    zrnb_percipitate,
    au_grating,
    twinned_nanowire,
    sample_with_g,
    mgo_nanocrystals,
)
import pytest


class TestLoadData:
    @pytest.mark.skip(reason="Downloading large file takes too long")
    def test_load_pdnip(self):
        s = pdnip_glass(allow_download=True)
        assert s.axes_manager.signal_shape == (128, 128)

    @pytest.mark.skip(reason="Downloading large file takes too long")
    def test_load_zrnb_percipitate(self):
        s = zrnb_percipitate(allow_download=True)
        assert s.axes_manager.signal_shape == (128, 128)

    def test_load_au_grating(self):
        s = au_grating(allow_download=True)
        assert s.axes_manager.signal_shape == (254, 254)

    @pytest.mark.skip(reason="Downloading large file takes too long")
    def test_load_twined_nanowire(self):
        s = twinned_nanowire(allow_download=True)
        assert s.axes_manager.signal_shape == (144, 144)

    @pytest.mark.skip(reason="Downloading large file takes too long")
    def test_sample_with_g(self):
        s = sample_with_g(allow_download=True)
        assert s.axes_manager.signal_shape == (256, 256)

    @pytest.mark.skip(reason="Downloading large file takes too long")
    def test_mgo(self):
        s = mgo_nanocrystals(allow_download=True)
        assert s.axes_manager.signal_shape == (144, 144)
