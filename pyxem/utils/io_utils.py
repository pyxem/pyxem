# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
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

# a lot of stuff depends on this, so we have to create it first

import glob
import logging
import os
import warnings

from hyperspy.io import load as hyperspyload
from hyperspy.io import load_with_reader

import numpy as np
import dask.array as da
from math import floor
from scipy.signal import find_peaks

from pyxem.signals.diffraction1d import Diffraction1D
from pyxem.signals.diffraction2d import Diffraction2D
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.vdf_image import VDFImage
from pyxem.signals.crystallographic_map import CrystallographicMap
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.indexation_results import TemplateMatchingResults

from pyxem.signals.diffraction1d import LazyDiffraction1D
from pyxem.signals.diffraction2d import LazyDiffraction2D
from pyxem.signals.electron_diffraction1d import LazyElectronDiffraction1D
from pyxem.signals.electron_diffraction2d import LazyElectronDiffraction2D

signal_dictionary = {'diffraction1d': Diffraction1D,
                     'diffraction2d': Diffraction2D,
                     'electron_diffraction1d': ElectronDiffraction1D,
                     'electron_diffraction2d': ElectronDiffraction2D,
                     'vdf_image': VDFImage,
                     'template_matching': TemplateMatchingResults,
                     'diffraction_vectors': DiffractionVectors,
                     'crystallographic_map': CrystallographicMap}

lazy_signal_dictionary = {'diffraction1d': LazyDiffraction1D,
                          'diffraction2d': LazyDiffraction2D,
                          'electron_diffraction1d': LazyElectronDiffraction1D,
                          'electron_diffraction2d': LazyElectronDiffraction2D}


def load(filename, lazy=False):
    """Load data into pyxem objects.

    Parameters
    ----------
    filename : str
        A single filename of a previously saved pyxem object. Other arguments may
        succeed, but will have fallen back on hyperspy load and warn accordingly
    lazy : bool
        If True the file will be opened lazily, i.e. without actually reading
        the data from the disk until required. Allows datasets much larger than
        available memory to be loaded.

    Returns
    -------
    s : Signal
        A pyxem Signal object containing loaded data.
    """
    s = hyperspyload(filename, lazy=lazy)
    if lazy:  # pragma: no cover
        try:
            s = lazy_signal_dictionary[s.metadata.Signal.signal_type](s)
        except KeyError:
            raise ValueError("Invalid signal_type in saved data for pyxem, "
                             "please use load_hspy for this data. ")
    else:
        try:
            s = signal_dictionary[s.metadata.Signal.signal_type](s)
        except KeyError:
            raise ValueError("Invalid signal_type in saved data for pyxem, "
                             "please use load_hspy for this data. ")

    return s


def load_hspy(filename, lazy=False, assign_to=None):
    """Wraps hyperspy.load to load various file formats and assigns suitable
    loaded data to user specified pyxem signals.

    Parameters
    ----------
    filename : str
        A single filename of a previously saved pyxem object. Other arguments may
        succeed, but will have fallen back on hyperspy load and warn accordingly
    lazy : bool
        If True the file will be opened lazily, i.e. without actually reading
        the data from the disk until required. Allows datasets much larger than
        available memory to be loaded.
    assign_to : str
        The signal class type the loaded data should be assigned to. If None,
        follows default hyperspy behaviour.

    Returns
    -------
    s : Signal
        A pyxem Signal object containing loaded data.
    """
    s = hyperspyload(filename, lazy=lazy)
    if assign_to:
        if lazy:  # pragma: no cover
            try:
                s = lazy_signal_dictionary[assign_to](s)
            except KeyError:
                raise ValueError("Invalid value of assign_to for lazy loading "
                                 "please specify a lazy pyXem signal or None. ")
        else:
            try:
                s = signal_dictionary[assign_to](s)
            except KeyError:
                raise ValueError("Invalid value of assign_to for loading "
                                 "please specify a pyXem signal or None. ")

    return s


def load_mib(mib_filename, reshape=True):
    """Read a .mib file using dask and return as LazyElectronDiffraction2D.

    Parameters
    ----------
    mib_filename : str
        The name of the .mib file to be read.
    reshape : boolean
        keywork argument to control reshaping of the stack (default is True).
        It attepmts to reshape using the flyback pixel.

    Returns
    -------
    data_pxm : pyxem.signals.LazyElectronDiffraction2D
                The metadata adds the following domains:
                General
                │   └── title =
                └── Signal
                    ├── binned = False
                    ├── exposure_time = 0.001
                    ├── flyback_times = [0.066, 0.071, 0.065, 0.017825]
                    ├── frames_number_skipped = 90
                    ├── scan_X = 256
                    └── signal_type = STEM
    """
    hdr_stuff = _parse_hdr(mib_filename)
    data = _read_mib(mib_filename, hdr_stuff)
    exp_times_list = _read_exposures(hdr_stuff, mib_filename)
    data_dict = _STEM_flag_dict(exp_times_list)

    if hdr_stuff['Assembly Size'] == '2x2':
        data = _add_crosses(data)

    data_pxm = LazyElectronDiffraction2D(data)

    # Tranferring dict info to metadata
    if data_dict['STEM_flag'] == 1:
        data_pxm.metadata.Signal.signal_type = 'STEM'
    else:
        data_pxm.metadata.Signal.signal_type = 'TEM'
    data_pxm.metadata.Signal.scan_X = data_dict['scan_X']
    data_pxm.metadata.Signal.exposure_time = data_dict['exposure time']
    data_pxm.metadata.Signal.frames_number_skipped = data_dict['number of frames_to_skip']
    data_pxm.metadata.Signal.flyback_times = data_dict['flyback_times']

    if data_pxm.metadata.Signal.signal_type is 'TEM' and data_pxm.metadata.Signal.exposure_time is not None:
        print('This mib file appears to be TEM data. The stack is returned with no reshaping.')
        return data_pxm
    try:
        if reshape:
            # If the exposure time info not appearing in the header bits use reshape_4DSTEM_SumFrames
            # to reshape otherwise use reshape_4DSTEM_FlyBack function
            if data_pxm.metadata.Signal.exposure_time is None:
                data_pxm = reshape_4DSTEM_SumFrames(data_pxm)
            else:
                data_pxm = reshape_4DSTEM_FlyBack(data_pxm)
    except TypeError:
        raise ValueError('Reshaping did not work. Get the stack by passing reshape=False')
    except ValueError:
        raise ValueError('Reshaping did not work. Get the stack by passing reshape=False')

    return data_pxm


def _manageHeader(fname):
    """Get necessary information from the header of the .mib file.

    Parameters
    ----------
    fname : str
        Filename for header file.

    Returns
    -------
    hdr : tuple
        (DataOffset,NChips,PixelDepthInFile,sensorLayout,Timestamp,shuttertime,bitdepth)

    Examples
    --------
    #Output for 6bit 256*256 data:
    #(768, 4, 'R64', '2x2', '2019-06-14 11:46:12.607836', 0.0002, 6)
    #Output for 12bit single frame nor RAW:
    #(768, 4, 'U16', '2x2', '2019-06-06 11:12:42.001309', 0.001, 12)

    """
    Header = str()
    with open(fname, 'rb') as input:
        aByte = input.read(1)
        Header += str(aByte.decode('ascii'))
        # This gets rid of the header
        while aByte and ord(aByte) != 0:

            aByte = input.read(1)
            Header += str(aByte.decode('ascii'))

    elements_in_header = Header.split(',')

    DataOffset = int(elements_in_header[2])

    NChips = int(elements_in_header[3])

    PixelDepthInFile = elements_in_header[6]
    sensorLayout = elements_in_header[7].strip()
    Timestamp = elements_in_header[9]
    shuttertime = float(elements_in_header[10])

    if PixelDepthInFile == 'R64':
        bitdepth = int(elements_in_header[18])  # RAW
    elif PixelDepthInFile == 'U16':
        bitdepth = 12
    elif PixelDepthInFile == 'U08':
        bitdepth = 6
    elif PixelDepthInFile == 'U32':
        bitdepth = 24

    hdr = (DataOffset, NChips, PixelDepthInFile, sensorLayout, Timestamp, shuttertime, bitdepth)

    return hdr


def _parse_hdr(fp):
    """Parse information from mib file header info from _manageHeader function.

    Parameters
    ----------
    fp : str
        Filepath to .mib file.

    Returns
    -------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.

    """
    hdr_info = {}

    read_hdr = _manageHeader(fp)

    # Set the array size of the chip

    if read_hdr[3] == '1x1':
        hdr_info['width'] = 256
        hdr_info['height'] = 256
    elif read_hdr[3] == '2x2':
        hdr_info['width'] = 512
        hdr_info['height'] = 512

    hdr_info['Assembly Size'] = read_hdr[3]

    # Set mib offset
    hdr_info['offset'] = read_hdr[0]
    # Set data-type
    hdr_info['data-type'] = 'unsigned'
    # Set data-length
    if read_hdr[6] == '1':
        # Binary data recorded as 8 bit numbers
        hdr_info['data-length'] = '8'
    else:
        # Changes 6 to 8 , 12 to 16 and 24 to 32 bit
        cd_int = int(read_hdr[6])
        hdr_info['data-length'] = str(int((cd_int + cd_int / 3)))

    hdr_info['Counter Depth (number)'] = int(read_hdr[6])
    if read_hdr[2] == 'R64':
        hdr_info['raw'] = 'R64'
    else:
        hdr_info['raw'] = 'MIB'
    # Set byte order
    hdr_info['byte-order'] = 'dont-care'
    # Set record by to stack of images
    hdr_info['record-by'] = 'image'

    # Set title to file name
    hdr_info['title'] = fp.split('.')[0]
    # Set time and date
    # Adding the try argument to accommodate the new hdr formatting as of April 2018
    try:
        year, month, day_time = read_hdr[4].split('-')
        day, time = day_time.split(' ')
        hdr_info['date'] = year + month + day
        hdr_info['time'] = time
    except BaseException:
        day, month, year_time = read_hdr[4].split('/')
        year, time = year_time.split(' ')
        hdr_info['date'] = year + month + day
        hdr_info['time'] = time

    hdr_info['data offset'] = read_hdr[0]

    return hdr_info


def _add_crosses(a):
    """ Adds 3 pixel buffer cross to quad chip data.

    Parameters
    ----------
    a : dask.array
        Stack of raw frames, prior to dimension reshaping, to insert
        3 pixel buffer cross into.

    Returns
    -------
    b : dask.array
        Stack of frames including 3 pixel buffer cross.
    """
    # Determine dimensions of raw frame data
    a_type = a.dtype
    a_shape = a.shape

    len_a_shape = len(a_shape)
    img_axes = len_a_shape - 2, len_a_shape - 1
    a_half = int(a_shape[img_axes[0]] / 2), int(a_shape[img_axes[1]] / 2)
    # Define 3 pixel wide cross of zeros to pad raw data
    z_array = da.zeros((a_shape[0], a_shape[1], 3), dtype=a_type)
    z_array2 = da.zeros((a_shape[0], 3, a_shape[img_axes[1]] + 3), dtype=a_type)
    # Insert blank cross into raw data

    b = da.concatenate((a[:, :, :a_half[1]], z_array, a[:, :, a_half[1]:]), axis=-1)

    b = da.concatenate((b[:, :a_half[0], :], z_array2, b[:, a_half[0]:, :]), axis=-2)

    return b


def _get_mib_depth(hdr_info, fp):
    """Determine the total number of frames based on .mib file size.

    Parameters
    ----------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.
    fp : filepath
        Path to .mib file.

    Returns
    -------
    depth : int
        Number of frames in the stack
    """
    # Define standard frame sizes for quad and single medipix chips
    if hdr_info['Assembly Size'] == '2x2':
        mib_file_size_dict = {
            '1': 33536,
            '6': 262912,
            '12': 525056,
            '24': 1049344,
        }
    if hdr_info['Assembly Size'] == '1x1':
        mib_file_size_dict = {
            '1': 8576,
            '6': 65920,
            '12': 131456,
            '24': 262528,
        }

    file_size = os.path.getsize(fp[:-3] + 'mib')
    if hdr_info['raw'] == 'R64':

        single_frame = mib_file_size_dict.get(str(hdr_info['Counter Depth (number)']))
        depth = int(file_size / single_frame)
    elif hdr_info['raw'] == 'MIB':
        if hdr_info['Counter Depth (number)'] == '1':
            # 1 bit and 6 bit non-raw frames have the same size
            single_frame = mib_file_size_dict.get('6')
            depth = int(file_size / single_frame)
        else:
            single_frame = mib_file_size_dict.get(str(hdr_info['Counter Depth (number)']))
            depth = int(file_size / single_frame)

    return depth


def _read_exposures(hdr_info, fp, pct_frames_to_read=0.1, mmap_mode='r'):
    """
    Looks into the frame times of the first 10 pct of the frames to see if they are
    all the same (TEM) or there is a flyback (4D-STEM).
    For this to work, the tick in the Merlin softeare to print exp time into header
    must be selected!

    Parameters
    -------------
    hdr_info: dict
        Output from _parse_hdr function
    fp: str
        MIB file name / path
    pct_frames_to_read : float
        Percentage of frames to read, default value 0.1
    mmap_mode: str
        Memmpa read mode - default is 'r'
    Returns
    ------------
    exp_time: list
        List of frame exposure times
    """
    width = hdr_info['width']
    height = hdr_info['height']
    depth = _get_mib_depth(hdr_info, fp)
    offset = hdr_info['offset']
    data_length = hdr_info['data-length']
    data_type = hdr_info['data-type']
    endian = hdr_info['byte-order']
    record_by = hdr_info['record-by']
    read_offset = 0

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian
    endian = '>'
    data_type += str(int(data_length))
    # uint1 not a valid dtype
    if data_type == 'uint1':
        data_type = 'uint8'
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    if data_length == '1':
        hdr_multiplier = 1
    else:
        hdr_multiplier = (int(data_length) / 8)**-1

    hdr_bits = int(hdr_info['data offset'] * hdr_multiplier)

    data = np.memmap(fp,
                     offset=read_offset,
                     dtype=data_type,
                     mode=mmap_mode)
    data = da.from_array(data)

    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        data = data.reshape(size)
    elif record_by == 'image':  # stack of images
        width_height = width * height

        size = (depth, height, width)

        # remove headers at the beginning of each frame and reshape

        if hdr_info['raw'] == 'R64':
            try:
                data = data.reshape(-1, width_height + hdr_bits)[:, 71:79]
                data = data[:, ]
                data_crop = data[:int(depth * pct_frames_to_read)]
                d = data_crop.compute()
                exp_time = []
                for line in range(d.shape[0]):
                    str_list = [chr(d[line][n]) for n in range(d.shape[1])]
                    exp_time.append(float(''.join(str_list)))
            except ValueError:
                print('Frame exposure times are not appearing in header!')

        else:
            try:
                if hdr_info['Counter Depth (number)'] == 1:
                    # RAW 1 bit data: the header bits are written as uint8 but the frames
                    # are binary and need to be unpacked as such.
                    data = data.reshape(-1, width_height / 8 + hdr_bits)[:, 71:79]
                else:
                    data = data.reshape(-1, width_height + hdr_bits)[:, 71:79]

                data = data[:, ]
                data_crop = data[:int(depth * pct_frames_to_read)]
                d = data_crop.compute()
                exp_time = []
                for line in range(d.shape[0]):
                    str_list = [chr(d[line][n]) for n in range(d.shape[1])]
                    exp_time.append(float(''.join(str_list)))
            except ValueError:
                print('Frame exposure times are not appearing in header!')

    elif record_by == 'dont-care':  # stack of images
        size = (height, width)
        data = data.reshape(size)
    return exp_time


def _STEM_flag_dict(exp_times_list):
    """Determines whether a .mib file contains STEM or TEM data and how many
    frames to skip due to triggering from a list of exposure times.

    Parameters
    ----------
    exp_times_list : list
        List of exposure times extracted from a .mib file.

    Returns
    -------
    output : dict
        Dictionary containing - STEM_flag, scan_X, exposure_time,
                                number_of_frames_to_skip, flyback_times
    """
    output = {}
    times_set = set(exp_times_list)
    # If single exposure times in header, treat as TEM data.
    if len(times_set) == 1:
        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = list(times_set)
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # In case exp times not appearing in header treat as TEM data
    elif len(times_set) == 0:

        output['STEM_flag'] = 0
        output['scan_X'] = None
        output['exposure time'] = None
        output['number of frames_to_skip'] = None
        output['flyback_times'] = None
    # Otherwise, treat as STEM data.
    else:
        STEM_flag = 1
        # Check that the smallest time is the majority of the values
        exp_time = max(times_set, key=exp_times_list.count)
        if exp_times_list.count(exp_time) < int(0.9 * len(exp_times_list)):
            print('Something wrong with the triggering!')
        peaks = [i for i, e in enumerate(exp_times_list) if e != exp_time]
        # Diff between consecutive elements of the array
        lines = np.ediff1d(peaks)

        if len(set(lines)) == 1:
            scan_X = lines[0]
            frames_to_skip = peaks[0]
        else:
            # Assuming the last element to be the line length
            scan_X = lines[-1]
            check = np.ravel(np.where(lines == scan_X, True, False))
            # Checking line lengths
            start_ind = np.where(check == False)[0][-1] + 2
            frames_to_skip = peaks[start_ind]

        flyback_times = list(times_set)
        flyback_times.remove(exp_time)
        output['STEM_flag'] = STEM_flag
        output['scan_X'] = scan_X
        output['exposure time'] = exp_time
        output['number of frames_to_skip'] = frames_to_skip
        output['flyback_times'] = flyback_times

    return output


def _read_mib(fp, hdr_info, mmap_mode='r'):
    """Read a raw .mib file using memory mapping where the array
    is stored on disk and not directly loaded, but may be treated
    like a numpy.ndarray.



    Parameters
    ----------
    fp: str
        Filepath of .mib file to be loaded.

    hdr_info: dict
        A dictionary containing the keywords as parsed by read_hdr
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode
        (see `numpy.memmap`).  The mode has no effect for pickled or
        zipped files.

    Returns
    -------
    data : numpy.memmap

    """

    reader_offset = 0

    width = hdr_info['width']
    height = hdr_info['height']

    offset = hdr_info['offset']
    data_length = hdr_info['data-length']
    data_type = hdr_info['data-type']
    endian = hdr_info['byte-order']
    record_by = hdr_info['record-by']

    depth = _get_mib_depth(hdr_info, fp)

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian
    endian = '>'

    data_type += str(int(data_length))
    if data_type == 'uint1':
        data_type = 'uint8'
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)
    if data_length == '1':
        hdr_multiplier = 1
    else:
        hdr_multiplier = (int(data_length) / 8)**-1

    hdr_bits = int(hdr_info['data offset'] * hdr_multiplier)

    data = np.memmap(fp,
                     offset=reader_offset,
                     dtype=data_type,
                     mode=mmap_mode)
    data = da.from_array(data)

    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        try:
            data = data.reshape(size)
        # in case of incomplete frame:
        except ValueError:
            if hdr_info['raw'] == 'R64':

                data = data.reshape(depth)

    elif record_by == 'image':  # stack of images
        width_height = width * height

        size = (depth, height, width)

        # remove headers at the beginning of each frame and reshape

        if hdr_info['Assembly Size'] == '2x2':
            if hdr_info['Counter Depth (number)'] == 1:
                # RAW 1 bit data: the header bits are written as uint8 but the frames
                # are binary and need to be unpacked as such.
                data = data.reshape(-1, width_height / 8 + hdr_bits)
                data = data[:, hdr_bits:]
                data = np.unpackbits(data)
                data = data.reshape(depth, width, height)
            else:
                data = data.reshape(-1, width_height + hdr_bits)[:, -width_height:].reshape(depth, width, height)
        elif hdr_info['Assembly Size'] == '1x1':
            data = data.reshape(-1, width_height + hdr_bits)[:, -width_height:].reshape(depth, width, height)
            data = data.reshape(depth, 256, 256)

        if hdr_info['raw'] == 'R64':
            if hdr_info['Counter Depth (number)'] == 24 or hdr_info['Counter Depth (number)'] == 12:
                COLS = 4

            if hdr_info['Counter Depth (number)'] == 1:
                COLS = 64

            if hdr_info['Counter Depth (number)'] == 6:
                COLS = 8

            data = data.reshape((depth * width_height))

            data = data.reshape(depth, height * (height // COLS), COLS)

            data = da.flip(data, 2)

            if hdr_info['Assembly Size'] == '2x2':

                data = data.reshape((depth * width_height))
                data = data.reshape(depth, 512 // 2, 512 * 2)

                det1 = data[:, :, 0:256]
                det2 = data[:, :, 256:512]
                det3 = data[:, :, 512:512 + 256]
                det4 = data[:, :, 512 + 256:]

                det3 = da.flip(det3, 2)
                det3 = da.flip(det3, 1)

                det4 = da.flip(det4, 2)
                det4 = da.flip(det4, 1)

                data = da.concatenate((da.concatenate((det1, det3), 1), da.concatenate((det2, det4), 1)), 2)

    elif record_by == 'dont-care':  # stack of images
        size = (height, width)
        data = data.reshape(size)

    return data


def reshape_4DSTEM_FlyBack(data):
    """Reshapes the lazy-imported frame stack to navigation dimensions determined
    based on stored exposure times.


    Parameters
    ----------
    data : hyperspy lazy Signal2D
        Lazy loaded electron diffraction data: <framenumbers | det_size, det_size>
        the data metadata contains flyback info as:
            ├── General
        │   └── title =
        └── Signal
            ├── binned = False
            ├── exposure_time = 0.001
            ├── flyback_times = [0.01826, 0.066, 0.065]
            ├── frames_number_skipped = 68
            ├── scan_X = 256
            └── signal_type = STEM

    Returns
    -------
    data_skip : pyxem.signals.LazyElectronDiffraction2D
        Reshaped electron diffraction data <scan_x, scan_y | det_size, det_size>
    """
    # Get detector size in pixels
    # detector size in pixels
    det_size = data.axes_manager[1].size
    # Read metadata
    skip_ind = data.metadata.Signal.frames_number_skipped
    line_len = data.metadata.Signal.scan_X

    n_lines = floor((data.data.shape[0] - skip_ind) / line_len)

    # Remove skipped frames
    data_skip = data.inav[skip_ind:skip_ind + (n_lines * line_len)]
    # Reshape signal
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()
    # Cropping the bright fly-back pixel
    data_skip = data_skip.inav[1:]

    return data_skip


def reshape_4DSTEM_SumFrames(data):
    """
    Reshapes the lazy-imported stack of dimensions: (xxxxxx|Det_X, Det_Y) to the correct scan pattern
    shape: (x, y | Det_X, Det_Y) when the frame exposure times are not in headre bits.
    It utilises the over-exposed fly-back frame to identify the start of the lines in the first 20
    lines of frames,checks line length consistancy and finds the number of frames to skip at the
    beginning (this number is printed out as string output).

    Parameters
    ----------
    data : pyxem LazyElectronDiffraction2D imported mib file with diensions of: framenumbers|Det_X, Det_Y

    Returns
    -------
    data_reshaped : reshaped data (x, y | Det_X, Det_Y)
    """
    # Assuming sacn_x, i.e. number of probe positions in a line is square root
    # of total number of frames
    scan_x = int(np.sqrt(data.axes_manager[0].size))
    # detector size in pixels
    det_size = data.axes_manager[1].size
    # crop the first ~20 lines
    data_crop = data.inav[0:20 * scan_x]
    data_crop_t = data_crop.T
    data_crop_t_sum = data_crop_t.sum()
    # summing over patterns
    intensity_array = data_crop_t_sum.data
    intensity_array = intensity_array.compute()
    peaks = find_peaks(intensity_array, distance=scan_x)
    # Difference between consecutive elements of the array
    lines = np.ediff1d(peaks[0])
    # Assuming the last element to be the line length
    line_len = lines[-1]
    if line_len != scan_x:
        raise ValueError('Fly_back does not correspond to correct line length!')
        #  Exits this routine and reshapes using scan size instead
    # Checking line lengths
    check = np.ravel(np.where(lines == line_len, True, False))
    # In case there is a False in there take the index of the last False
    if ~np.all(check):
        start_ind = np.where(check == False)[0][-1] + 2
        # Adding 2 - instead of 1 -  to be sure scan is OK
        skip_ind = peaks[0][start_ind]
     # In case they are all True take the index of the first True
    else:
         # number of frames to skip at the beginning
        skip_ind = peaks[0][0]
    # Number of lines
    n_lines = floor((data.data.shape[0] - skip_ind) / line_len)
    # with the skipped frames removed
    data_skip = data.inav[skip_ind:skip_ind + (n_lines * line_len)]
    data_skip.data = data_skip.data.reshape(n_lines, line_len, det_size, det_size)
    data_skip.axes_manager._axes.insert(0, data_skip.axes_manager[0].copy())
    data_skip.get_dimensions_from_data()
    print('Reshaping using the frame intensity sums of the first 20 lines')
    print('Number of frames skipped at the beginning: ', skip_ind)
    # Croppimg the flyaback pixel at the start
    data_skip = data_skip.inav[1:]
    return data_skip
