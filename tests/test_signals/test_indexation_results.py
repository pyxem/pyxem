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

import numpy as np
import pytest
from pyxem.signals.indexation_results import TemplateMatchingResults, VectorMatchingResults
from tests.test_utils.test_indexation_utils import sp_vector_match_result, dp_vector_match_result


def test_get_crystalographic_map(dp_vector_match_result, sp_vector_match_result):
    # Assertion free test, as the tests in test_indexation_utils do the heavy
    # lifting
    results = np.vstack((dp_vector_match_result[0], sp_vector_match_result[0]))
    results = TemplateMatchingResults(results)
    results.get_crystallographic_map()
    return 0
