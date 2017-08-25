import h5py
import logging
from hyperspy.io_plugins import emd
from hyperspy.io import load_with_reader
from hyperspy.io import load
from fpd_data_processing.pixelated_stem_class import (
        PixelatedSTEM, DPCBaseSignal, DPCSignal1D, DPCSignal2D)


def _fpd_checker(filename, attr_substring='fpd_version'):
    hdf5_file = h5py.File(filename)
    for attr in hdf5_file.attrs:
        if attr_substring in attr:
            return(True)
    return(False)


def _hspy_checker(filename, attr_substring='fpd_version'):
    hdf5_file = h5py.File(filename)
    for attr in hdf5_file.attrs:
        if 'file_format' in attr:
            if hdf5_file.attrs['file_format'] == 'HyperSpy':
                return(True)
    return(False)


def _load_fpd_emd_file(filename):
    logging.basicConfig(level=logging.ERROR)
    s_list = load_with_reader(filename, reader=emd)
    logging.basicConfig(level=logging.WARNING)
    temp_s = None
    longest_dims = 0
    for s in s_list:
        if len(s.data.shape) > longest_dims:
            longest_dims = len(s.data.shape)
    if longest_dims < 4:
        raise ValueError("Pixelated dataset not found")
    for s in s_list:
        if len(s.data.shape) == longest_dims:
            temp_s = s
            break
    if longest_dims == 4:
        s = temp_s.transpose(signal_axes=(0, 1))
    elif longest_dims == 5:
        s = temp_s.isig[:,:,0,:,:].transpose(signal_axes=(0, 1))
    else:
        raise Exception(
                "Pixelated dataset not found")
    return(s)


def load_fpd_signal(filename):
    if _fpd_checker(filename, attr_substring='fpd_version'):
        s = _load_fpd_emd_file(filename)
    elif _hspy_checker(filename, attr_substring='HyperSpy'):
        s = load(filename)
    else:
        raise IOError("File " + str(filename) + " not recognised")
    s_new = PixelatedSTEM(s.data)
    for i in range(len(s.axes_manager.shape)):
        s_new.axes_manager[i].offset = s.axes_manager[i].offset
        s_new.axes_manager[i].scale = s.axes_manager[i].scale
        s_new.axes_manager[i].name = s.axes_manager[i].name
        s_new.axes_manager[i].units = s.axes_manager[i].units
    s_new.metadata = s.metadata.deepcopy()
    return PixelatedSTEM(s.data)


def load_dpc_signal(filename):
    s = load(filename)
    if s.axes_manager.navigation_shape != (2,):
        raise Exception(
                "DPC signal needs to have 1 navigation "
                "dimension with a size of 2.")
    if s.axes_manager.signal_dimension == 0:
        s = DPCBaseSignal(load(filename)).T
    elif s.axes_manager.signal_dimension == 1:
        s = DPCSignal1D(load(filename))
    elif s.axes_manager.signal_dimension == 2:
        s = DPCSignal2D(load(filename))
    else:
        raise NotImplementedError(
                "DPC signals only support 0, 1 and 2 signal dimensions")
    return(s)
