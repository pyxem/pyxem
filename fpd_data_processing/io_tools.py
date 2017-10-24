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
    if h5py.is_hdf5(filename):
        hdf5_file = h5py.File(filename)
        for attr in hdf5_file.attrs:
            if attr_substring in attr:
                return(True)
    return(False)


def _hspy_checker(filename, attr_substring='fpd_version'):
    if h5py.is_hdf5(filename):
        hdf5_file = h5py.File(filename)
        for attr in hdf5_file.attrs:
            if 'file_format' in attr:
                if hdf5_file.attrs['file_format'] == 'HyperSpy':
                    return(True)
    return(False)


def _load_lazy_fpd_file(filename, chunk_size=(16, 16)):
    f = h5py.File(filename)
    if 'fpd_expt' in f:
        data = f['/fpd_expt/fpd_data/data']
        if len(data.shape) == 5:
            chunks = (
                    chunk_size[0], chunk_size[1],
                    1, data.shape[-2], data.shape[-1])
            data_lazy = da.from_array(data, chunks=chunks)[:, :, 0, :, :]
        elif len(data.shape) == 4:
            chunks = (
                    chunk_size[0], chunk_size[1],
                    data.shape[-2], data.shape[-1])
            data_lazy = da.from_array(data, chunks=chunks)[:, :, :, :]
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
        s = temp_s.isig[:, :, 0, :, :].transpose(signal_axes=(0, 1))
    else:
        raise Exception(
                "Pixelated dataset not found")

    s_new = _signal2d_to_pixelated_stem(s)
    return(s_new)


def _load_other_file(filename, lazy=False):
    s = load(filename, lazy=lazy)
    s_new = _signal2d_to_pixelated_stem(s)
    return s_new


def _signal2d_to_pixelated_stem(s):
    s_new = PixelatedSTEM(s.data)
    for i in range(len(s.axes_manager.shape)):
        s_new.axes_manager[i].offset = s.axes_manager[i].offset
        s_new.axes_manager[i].scale = s.axes_manager[i].scale
        s_new.axes_manager[i].name = s.axes_manager[i].name
        s_new.axes_manager[i].units = s.axes_manager[i].units
    s_new.metadata = s.metadata.deepcopy()
    return s_new


def load_fpd_signal(filename, lazy=False, chunk_size=(16, 16)):
    """
    Parameters
    ----------
    filename : string
    lazy : bool, default False
    chunk_size : tuple, default (16, 16)
        Used if Lazy is True. Sets the chunk size of the signal in the
        navigation dimension. Higher number will potentially make the
        calculations be faster, but use more memory.
    """
    if _fpd_checker(filename, attr_substring='fpd_version'):
        if lazy:
            s = _load_lazy_fpd_file(filename, chunk_size=chunk_size)
        else:
            s = _load_fpd_emd_file(filename)
    elif _hspy_checker(filename, attr_substring='HyperSpy'):
        s = _load_other_file(filename, lazy=lazy)
    else:
        # Attempt to load non-fpd and non-HyperSpy signal
        s = _load_other_file(filename, lazy=lazy)
    return s


def load_dpc_signal(filename):
    """Load a differential phase contrast style signal.

    This function can both files saved directly using HyperSpy,
    and saved using this library. The only requirement is that
    the signal has one navigation dimension, with this one dimension
    having a size of two. The first navigation index is the x-shift,
    while the second is the y-shift.
    The signal dimension contains the spatial dimension(s), i.e. the
    probe positions.

    The return signal depends on the dimensions of the input file:
    - If two signal dimensions: DPCSignal2D
    - If one signal dimension: DPCSignal1D
    - If zero signal dimension: DPCBaseSignal

    Parameters
    ----------
    filename : string

    Returns
    -------
    dpc_signal : DPCBaseSignal, DPCSignal1D, DPCSignal2D
        The type of return signal depends on the signal dimensions of the
        input file.

    Examples
    --------
    >>> import fpd_data_processing.api as fp
    >>> import numpy as np
    >>> s = fp.DPCSignal2D(np.random.random((2, 90, 50)))
    >>> s.save("test_dpc_signal2d.hspy")
    >>> s_dpc = fp.load_dpc_signal("test_dpc_signal2d.hspy")
    >>> s_dpc
    <DPCSignal2D, title: , dimensions: (2|50, 90)>
    >>> s_dpc.plot()

    Saving a HyperSpy signal

    >>> import hyperspy.api as hs
    >>> s = hs.signals.Signal1D(np.random.random((2, 10)))
    >>> s.save("test_dpc_signal1d.hspy")
    >>> s_dpc_1d = fp.load_dpc_signal("test_dpc_signal1d.hspy")
    >>> s_dpc_1d
    <DPCSignal1D, title: , dimensions: (2|10)>

    """
    s = load(filename)
    if s.axes_manager.navigation_shape != (2,):
        raise Exception(
                "DPC signal needs to have 1 navigation "
                "dimension with a size of 2.")
    if s.axes_manager.signal_dimension == 0:
        s_out = DPCBaseSignal(s).T
    elif s.axes_manager.signal_dimension == 1:
        s_out = DPCSignal1D(s)
    elif s.axes_manager.signal_dimension == 2:
        s_out = DPCSignal2D(s)
    else:
        raise NotImplementedError(
                "DPC signals only support 0, 1 and 2 signal dimensions")
    s_out.metadata = s.metadata.deepcopy()
    s_out.axes_manager = s.axes_manager.deepcopy()
    return s_out
