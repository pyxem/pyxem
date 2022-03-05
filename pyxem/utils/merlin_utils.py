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

import io

import numpy as np


def parse_header_mib(filename):
    """
    Parse the header of the first frame of a mib file

    Parameters
    ----------
    filename : str
        Filename of the mib file.

    Returns
    -------
    dict
        Header of the first frame.

    """
    with io.open(file=filename, mode="r", encoding="ascii", errors='ignore') as f:
        header_str = f.read(1024)

    header = header_str.split(',')
    header_dict = {}

    header_dict['header_dictID'] = header[0]
    header_dict['AcquisitionSequenceNumber'] = header[1]
    header_dict['DataOffset'] = np.uint32(header[2])
    header_dict['NumberOfChips'] = np.uint32(header[3])
    header_dict['PixelDimensionX'] = np.uint32(header[4])
    header_dict['PixelDimensionY'] = np.uint32(header[5])
    header_dict['PixelDepth'] = header[6]
    header_dict['SensorLayout'] = header[7][3:]
    header_dict['ChipSelect'] = header[8]
    header_dict['TimeStamp'] = header[9]
    header_dict['ShutterTime'] = float(header[10])
    header_dict['Counter'] = np.uint32(header[11])
    header_dict['ColourMode'] = np.uint32(header[12])
    header_dict['GainMode'] = np.uint32(header[13])
    header_dict['Thresholds'] = np.array([float(header[14+i]) for i in range(8)])
    header_dict['DACs'] = {}
    header_dict['DACs']['Format'] = header[22]
    header_dict['DACs']['Thresh0'] = np.uint16(header[23])
    header_dict['DACs']['Thresh1'] = np.uint16(header[24])
    header_dict['DACs']['Thresh2'] = np.uint16(header[25])
    header_dict['DACs']['Thresh3'] = np.uint16(header[26])
    header_dict['DACs']['Thresh4'] = np.uint16(header[27])
    header_dict['DACs']['Thresh5'] = np.uint16(header[28])
    header_dict['DACs']['Thresh6'] = np.uint16(header[29])
    header_dict['DACs']['Thresh7'] = np.uint16(header[30])
    header_dict['DACs']['Preamp'] = np.uint8(header[31])
    header_dict['DACs']['Ikrum'] = np.uint8(header[32])
    header_dict['DACs']['Shaper'] = np.uint8(header[33])
    header_dict['DACs']['Disc'] = np.uint8(header[34])
    header_dict['DACs']['DiscLS'] = np.uint8(header[35])
    header_dict['DACs']['ShaperTest'] = np.uint8(header[36])
    header_dict['DACs']['DACDiscL'] = np.uint8(header[37])
    header_dict['DACs']['DACTest'] = np.uint8(header[38])
    header_dict['DACs']['DACDISCH'] = np.uint8(header[39])
    header_dict['DACs']['Delay'] = np.uint8(header[40])
    header_dict['DACs']['TPBuffIn'] = np.uint8(header[41])
    header_dict['DACs']['TPBuffOut'] = np.uint8(header[42])
    header_dict['DACs']['RPZ'] = np.uint8(header[43])
    header_dict['DACs']['GND'] = np.uint8(header[44])
    header_dict['DACs']['TPRef'] = np.uint8(header[45])
    header_dict['DACs']['FBK'] = np.uint8(header[46])
    header_dict['DACs']['Cas'] = np.uint8(header[47])
    header_dict['DACs']['TPRefA'] = np.uint16(header[48])
    header_dict['DACs']['TPRefB'] = np.uint16(header[49])
    header_dict['ExtID'] = header[50]
    header_dict['ExtTimeStamp'] = header[51]
    header_dict['ExtID'] = float(header[52][:-2])
    header_dict['ExtCounterDepth'] = np.uint8(header[53])

    return header_dict


def copy_header_mib(filename, data_offset, output_filename):
    """
    Read the header of the first frame of a mib file and write it to disk.
    Useful to extract header from experimental mib file in order to create
    synthetic mib file to be used by the test suite.

    Parameters
    ----------
    filename : str
        File name of an experimental mib file.
    data_offset : int
        The offset of the data (= end of the header) in bytes.
    output_filename : str
        File name of the header filename.

    Returns
    -------
    raw_header : byte string
        The header as a byte string.

    """

    with open(filename, 'rb') as f:
        raw_header = f.read(data_offset)

    with open(output_filename, "wb") as f:
        f.write(raw_header)

    return raw_header


def create_4DSTEM(filename, header_filename, frame_size, frame_number, dtype):
    """
    Create a 4DSTEM mib file.

    Parameters
    ----------
    filename : str
        Name of the output file.
    header_filename : str
        Name of the file containing the header.
    frame_size : tuple
        Size of the frame.
    frame_number : int
        Number of frames
    dtype : python or numpy type
        Data dtype of the numpy array which will be write on disk using the
        ``numpy.ndarray.tobytes`` method.

    Returns
    -------
    None.

    """
    # header as byte string
    with open(header_filename, 'rb') as f:
        header = f.read()

    # some data
    data = np.arange(np.prod(frame_size)).reshape(frame_size) / 5
    data = data.astype(dtype)

    with open(filename, "wb") as f:
        for _ in range(frame_number):
            f.write(header)
            f.write(data.tobytes())
