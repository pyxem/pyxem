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

import os
from tempfile import TemporaryDirectory
import pytest
import numpy
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pyxem as pxm


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    plt.ioff()
    tmp_dir = TemporaryDirectory()
    hs.preferences.General.show_progressbar = False
    org_dir = os.getcwd()
    os.chdir(tmp_dir.name)
    yield
    os.chdir(org_dir)
    tmp_dir.cleanup()
    plt.close("all")


@pytest.fixture(autouse=True)
def add_np_am(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["hs"] = hs
    doctest_namespace["pxm"] = pxm
