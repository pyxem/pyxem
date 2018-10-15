# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
import numpy as np

from pyxem.signals.diffraction_variance import DiffractionVariance
from pyxem.signals.diffraction_variance import ImageVariance
from pyxem.signals.variance_profile import DiffractionVarianceProfile

@pytest.fixture
def diffraction_variance(diffraction_pattern):
    return DiffractionVariance(diffraction_pattern)

class TestDiffractionVariance:

    def test_get_diffraction_variance_signal(self,
                        diffraction_pattern):
        difvar = DiffractionVariance(diffraction_pattern)
        assert isinstance(difvar,DiffractionVariance)

    def test_difvar_radial_profile(self,
                        diffraction_pattern):
        rp = DiffractionVarianceProfile(diffraction_pattern)
        assert isinstance(rp,DiffractionVarianceProfile)

    @pytest.mark.parametrize('dqe', [
        0.5,
        0.6
    ])
    def test_renormalize(self,
                        diffraction_pattern,
                        dqe):
        difvar = DiffractionVariance(diffraction_pattern)
        difvar.renormalize(dqe)
        assert isinstance(difvar,DiffractionVariance)


class TestImageVariance:

    def test_get_image_variance_signal(self,
                        diffraction_pattern):
        imvar = ImageVariance(diffraction_pattern)
        assert isinstance(imvar,ImageVariance)
