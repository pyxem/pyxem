# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
from pyxem.signals.reduced_intensity_profile import ReducedIntensityProfile
from pyxem.components.scattering_fit_component import ScatteringFitComponent


def test_scattering_component_init_lobato():
    elements = ['Cu']
    fracs = [1]
    ref = ScatteringFitComponent(elements, fracs, N=1., C=0., type='lobato')
    assert isinstance(ref, ScatteringFitComponent)
    return


def test_scattering_component_init_xtables():
    elements = ['Cu']
    fracs = [1]
    ref = ScatteringFitComponent(elements, fracs, N=1., C=0., type='xtables')
    assert isinstance(ref, ScatteringFitComponent)
    return


@pytest.mark.xfail
def test_scattering_component_init_not_implemented():
    elements = ['Cu']
    fracs = [1]
    ref = ScatteringFitComponent(elements, fracs, N=1., C=0., type='nope')


@pytest.fixture(params=[
    np.array([4., 3., 2., 2., 1., 1., 1., 0.])
])
def ri_model(request):
    ri = ReducedIntensityProfile(request.param)
    m = ri.create_model()
    return m


def test_function_lobato(ri_model):
    elements = ['Cu']
    fracs = [1]
    sc_component = ScatteringFitComponent(elements, fracs, N=1., C=0., type='lobato')
    ri_model.append(sc_component)
    ri_model.fit()
    return


def test_function_xtables(ri_model):
    elements = ['Cu']
    fracs = [1]
    sc_component = ScatteringFitComponent(elements, fracs, N=1., C=0., type='xtables')
    ri_model.append(sc_component)
    ri_model.fit()
    return