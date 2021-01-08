# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

import pytest
from pyxem.detectors.medipix_515x515 import Medipix515x515Detector
from pyFAI.detectors import Detector
import numpy as np


def test_medipix_515x515_init():
    detector = Medipix515x515Detector()
    assert isinstance(detector, Detector)


def test_medipix_515x515_mask():
    detector = Medipix515x515Detector()
    mask = detector.calc_mask()
    mask_check = np.zeros((515, 515))
    mask_check[255:260, :] = 1
    mask_check[:, 255:260] = 1
    assert np.array_equal(mask, mask_check)
