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


def test_import():
    import pyxem

    for obj_name in pyxem.__all__:
        getattr(pyxem, obj_name)


def test_import_signals():
    import pyxem.signals

    for obj_name in pyxem.signals.__all__:
        getattr(pyxem.signals, obj_name)


def test_import_components():
    import pyxem.components

    for obj_name in pyxem.components.__all__:
        getattr(pyxem.components, obj_name)


def test_import_utils():
    import pyxem.utils

    for obj_name in pyxem.utils.__all__:
        getattr(pyxem.utils, obj_name)
