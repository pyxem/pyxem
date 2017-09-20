import h5py
import logging
import dask.array as da
from hyperspy.io_plugins import emd
from hyperspy.io import load_with_reader
from hyperspy.io import load
from fpd_data_processing.pixelated_stem_class import (
        PixelatedSTEM, DPCBaseSignal, DPCSignal1D, DPCSignal2D,
        LazyPixelatedSTEM)


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


def _load_lazy_fpd_file(filename):
    f = h5py.File(filename)
    if 'fpd_expt' in f:
        data = f['/fpd_expt/fpd_data/data']
        if len(data.shape) == 5:
            chunks = (1, 1, 1, data.shape[-2], data.shape[-1])
            data_lazy = da.from_array(data, chunks=chunks)[:,:,0,:,:]
        elif len(data.shape) == 4:
            chunks = (1, 1, data.shape[-2], data.shape[-1])
            data_lazy = da.from_array(data, chunks=chunks)[:,:,:,:]
        else:
            raise IOError(
                "Pixelated dataset does not have correct dimensions")

        s = LazyPixelatedSTEM(data_lazy)
        return(s)
    else:
        raise IOError("Pixelated dataset not found")


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


def load_fpd_signal(filename, lazy=False):
    if _fpd_checker(filename, attr_substring='fpd_version'):
        if lazy:
            s_new = _load_lazy_fpd_file(filename)
        else:
            s = _load_fpd_emd_file(filename)
    elif _hspy_checker(filename, attr_substring='HyperSpy'):
        s = load(filename, lazy=lazy)
    else:
        raise IOError("File " + str(filename) + " not recognised")
    if not lazy:
        s_new = PixelatedSTEM(s.data)
        for i in range(len(s.axes_manager.shape)):
            s_new.axes_manager[i].offset = s.axes_manager[i].offset
            s_new.axes_manager[i].scale = s.axes_manager[i].scale
            s_new.axes_manager[i].name = s.axes_manager[i].name
            s_new.axes_manager[i].units = s.axes_manager[i].units
        s_new.metadata = s.metadata.deepcopy()
    return s_new


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
