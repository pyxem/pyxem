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
from pyxem.utils.ri_utils import *
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D


def test_as_signal_generation():

    N = ElectronDiffraction2D([[np.array(1.)]])
    C = ElectronDiffraction2D([[np.array(0.)]])
    elements = ['Cu']
    fracs = [1]
    s_size = 10
    s_scale = 0.1
    types = ['lobato', 'xtables', 'not_implemented']

    for type in types:
        if type == 'lobato':
            signal, normalisation = scattering_to_signal(elements, fracs, N, C,
                                                         s_size, s_scale, type)
            expected_signal = np.array([[[31.371201, 21.08535486, 10.62320925,
                                          5.89629809, 3.51507336, 2.15565751, 1.34986551,
                                          0.8664032, 0.57201346, 0.38888391]]])
            expected_normalisation = np.array([[[5.601, 4.59187923, 3.2593265,
                                                 2.42822942, 1.87485289, 1.46821576, 1.16183713,
                                                 0.93080782, 0.75631572, 0.62360558]]])
            assert np.allclose(signal, expected_signal)
            assert np.allclose(normalisation, expected_normalisation)

        elif type == 'xtables':
            signal, normalisation = scattering_to_signal(elements, fracs, N, C,
                                                         s_size, s_scale, type)
            expected_signal = np.array([[[31.23021456, 21.12038612, 10.61694231,
                                          5.9564419, 3.47051602, 2.11850579, 1.36598179,
                                          0.90445736, 0.60043364, 0.39823201]]])
            expected_normalisation = np.array([[[5.5884, 4.59569213, 3.25836498,
                                                 2.44058229, 1.8629321, 1.45550877, 1.16875224,
                                                 0.95102963, 0.77487653, 0.63105627]]])
            assert np.allclose(signal, expected_signal)
            assert np.allclose(normalisation, expected_normalisation)
        else:
            # expect error
            with pytest.raises(NotImplementedError):
                signal, normalisation = scattering_to_signal(elements, fracs, N, C,
                                                             s_size, s_scale, type)

    return
