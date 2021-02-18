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

from pyxem.utils.signal import select_method_from_method_dict


@pytest.fixture()
def method_dict():
    return {"dummy_choice": select_method_from_method_dict}


def test_select_fake_method_from_method_dict(method_dict):
    with pytest.raises(NotImplementedError):
        _ = select_method_from_method_dict("fake_choice", method_dict, print_help=True)


def test_select_method_from_method_dict_print_help(method_dict):
    method_dict = {"dummy_choice": select_method_from_method_dict}
    _ = select_method_from_method_dict("dummy_choice", method_dict, print_help=True)
