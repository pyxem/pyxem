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

import h5py
import logging
import numpy as np
import dask.array as da
from hyperspy.io_plugins import emd
from hyperspy.io import load_with_reader
from hyperspy.io import load

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.electron_diffraction2d import LazyElectronDiffraction2D


def _get_dtype_from_header_string(header_string):
    header_split_list = header_string.split(",")
    dtype_string = header_split_list[6]
    if dtype_string == "U08":
        dtype = ">u1"
    elif dtype_string == "U16":
        dtype = ">u2"
    elif dtype_string == "U32":
        dtype = ">u4"
    else:
        print("dtype {0} not recognized, trying unsigned 16 bit".format(dtype_string))
        dtype = ">u2"
    return dtype


def _get_detector_pixel_size(header_string):
    header_split_list = header_string.split(",")
    det_x_string = header_split_list[4]
    det_y_string = header_split_list[5]
    try:
        det_x = int(det_x_string)
        det_y = int(det_y_string)
    except NameError:
        print(
            "detector size strings {0} and {1} not recognized, "
            "trying 256 x 256".format(det_x_string, det_y_string)
        )
        det_x, det_y = 256, 256
    if det_x == 256:
        det_x_value = det_x
    elif det_x == 512:
        det_x_value = det_x
    else:
        print("detector x size {0} not recognized, trying 256".format(det_x))
        det_x_value = 256
    if det_y == 256:
        det_y_value = det_y
    elif det_y == 512:
        det_y_value = det_y
    else:
        print("detector y size {0} not recognized, trying 256".format(det_y))
        det_y_value = 256
    return (det_x_value, det_y_value)


def _fpd_checker(filename, attr_substring="fpd_version"):
    if h5py.is_hdf5(filename):
        hdf5_file = h5py.File(filename, mode="r")
        for attr in hdf5_file.attrs:
            if attr_substring in attr:
                return True
    return False


def _hspy_checker(filename, attr_substring="fpd_version"):
    if h5py.is_hdf5(filename):
        hdf5_file = h5py.File(filename, mode="r")
        for attr in hdf5_file.attrs:
            if "file_format" in attr:
                if hdf5_file.attrs["file_format"] == "HyperSpy":
                    return True
    return False


def _load_fpd_sum_im(filename):
    s = load_with_reader(filename, reader=emd, dataset_name="/fpd_expt/fpd_sum_im")
    if len(s.axes_manager.shape) == 3:
        s = s.isig[0, :, :]
    return s


def _load_fpd_sum_dif(filename):
    s = load_with_reader(filename, reader=emd, dataset_name="/fpd_expt/fpd_sum_dif")
    if len(s.axes_manager.shape) == 3:
        s = s.isig[:, :, 0]
    return s


def _load_fpd_emd_file(filename, lazy=False):
    s = load_with_reader(
        filename, reader=emd, lazy=lazy, dataset_name="/fpd_expt/fpd_data"
    )
    if len(s.axes_manager.shape) == 5:
        s = s.isig[:, :, 0, :, :]
    s = s.transpose(signal_axes=(0, 1))
    s._lazy = lazy
    s_new = signal_to_pixelated_stem(s)
    return s_new


def _load_other_file(filename, lazy=False):
    s = load(filename, lazy=lazy)
    s_new = signal_to_pixelated_stem(s)
    return s_new


def _copy_axes_ps_to_dpc(s_ps, s_dpc):
    if s_ps.axes_manager.navigation_dimension > 2:
        raise ValueError(
            "s_ps can not have more than 2 navigation dimensions, "
            "not {0}".format(s_ps.axes_manager.navigation_dimension)
        )
    if s_ps.axes_manager.navigation_shape != s_dpc.axes_manager.signal_shape:
        raise ValueError(
            "s_ps navigation shape {0}, must be the same "
            "as s_dpc signal shape {1}".format(
                s_ps.axes_manager.navigation_shape, s_dpc.axes_manager.signal_shape
            )
        )
    ps_a_list = s_ps.axes_manager.navigation_axes
    dp_a_list = s_dpc.axes_manager.signal_axes
    for ps_a, dp_a in zip(ps_a_list, dp_a_list):
        dp_a.offset = ps_a.offset
        dp_a.scale = ps_a.scale
        dp_a.units = ps_a.units
        dp_a.name = ps_a.name


def load_ps_signal(filename, lazy=False, chunk_size=None, navigation_signal=None):
    """
    Parameters
    ----------
    filename : string
    lazy : bool, default False
    chunk_size : tuple, optional
        Used if Lazy is True. Sets the chunk size of the signal.
        If it is not specified, the file chunking will be used.
        Higher number will potentially make the calculations be faster,
        but use more memory.
    navigation_signal : Signal2D

    """
    if _fpd_checker(filename, attr_substring="fpd_version"):
        s = _load_fpd_emd_file(filename, lazy=lazy)
    elif _hspy_checker(filename, attr_substring="HyperSpy"):
        s = _load_other_file(filename, lazy=lazy)
    else:
        # Attempt to load non-fpd and non-HyperSpy signal
        s = _load_other_file(filename, lazy=lazy)
    if navigation_signal is None:
        try:
            s_nav = _load_fpd_sum_im(filename)
            s.navigation_signal = s_nav
        except IOError:
            logging.debug("Nav signal not found in {0}".format(filename))
            s.navigation_signal = None
        except ValueError:
            logging.debug("Nav signal in {0}: wrong shape".format(filename))
            s.navigation_signal = None
    else:
        nav_im_shape = navigation_signal.axes_manager.signal_shape
        nav_ax_shape = s.axes_manager.navigation_shape
        if nav_im_shape == nav_ax_shape:
            s.navigation_signal = navigation_signal
        else:
            raise ValueError(
                "navigation_signal does not have the same shape ({0}) as "
                "the signal's navigation shape ({1})".format(nav_im_shape, nav_ax_shape)
            )
    if lazy:
        if chunk_size is not None:
            s.data = s.data.rechunk(chunks=chunk_size)
    return s
