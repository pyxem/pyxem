# -*- coding: utf-8 -*-
# Copyright 2021 The pyXem developers
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
import numpy as np
import tempfile

from pyxem.utils.merlin_utils import create_4DSTEM
from pyxem import load_mib


MERLIN_DATA_PATH = os.path.join(os.path.dirname(__file__), 'merlin_data')
TMP_FOLDER = tempfile.gettempdir()


"""
from pyxem.utils.merlin_utils import parse_header_mib, copy_header_mib

# Generate a single image mib file
filename = 'path_to_filename_to_extract_header'
header = parse_header_mib(filename)

output_filename = os.path.join(MERLIN_DATA_PATH, 'single_frame_quad.mib_header')
raw_header = copy_header_mib(filename, data_offset=header['DataOffset'],
                             output_filename=output_filename)
"""

"""
from pyxem.utils.merlin_utils import parse_header_mib, copy_header_mib

filename = 'path_to_filename_to_extract_header'
header = parse_header_mib(filename)
output_filename = os.path.join(MERLIN_DATA_PATH, '2x16_quad.mib_header')
raw_header = copy_header_mib(filename, data_offset=header['DataOffset'],
                             output_filename=output_filename)
"""


def test_single_frame():
    single_frame_fname = os.path.join(TMP_FOLDER, 'single_frame_quad.mib')
    header_filename = os.path.join(MERLIN_DATA_PATH, 'single_frame_quad.mib_header')
    frame_size = (512, 512)
    dtype = '>u2'
    create_4DSTEM(single_frame_fname, header_filename, frame_size, 1, dtype)

    s = load_mib(single_frame_fname)
    s.compute(close_file=True)

    data = np.arange(np.prod(frame_size)).reshape(frame_size) / 5

    np.testing.assert_allclose(s.data[0], data.astype(dtype))


def test_2x16():
    _2x16_fname = os.path.join(TMP_FOLDER, '2x16_quad.mib')
    header_filename = os.path.join(MERLIN_DATA_PATH, '2x16_quad.mib_header')
    frame_size = (512, 512)
    dtype = '>u2'
    create_4DSTEM(_2x16_fname, header_filename, frame_size, 2*16, dtype)

    s = load_mib(_2x16_fname)
    s.compute(close_file=True)

    data = np.arange(np.prod(frame_size)).reshape(frame_size) / 5

    np.testing.assert_allclose(s.data[0], data.astype(dtype))
