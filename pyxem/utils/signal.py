# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

"""Utils for signal processing."""

# This code is not used internally and so can safely be deleted at the end of the deprecation period

from pyxem.utils._deprecated import deprecated
import dask.array as da


@deprecated(since="0.18.0", removal="0.19.0")
def select_method_from_method_dict(method, method_dict, print_help=True, **kwargs):
    """Streamlines the selection of utils to be mapped in class methods.

    Parameters
    ----------
    method : str
        The key to method_dict for the chosen method
    method_dict : dict
        dictionary with strings as keys and functions as values
    kwargs : dict
        Parameters for the method, if empty help is return
    print_help : bool
        If True: Prints information about the chosen method.

    Returns
    -------
    method_function :
        The utility function that corresponds the given method string, unless
        kwargs is empty, in which case the help for the utility function is
        returned.
    """

    if method not in method_dict:
        raise NotImplementedError(
            "The method `{}` is not implemented. "
            "See documentation for available "
            "implementations.".format(method)
        )
    elif print_help and not kwargs:
        help(method_dict[method])

    return method_dict[method]


@deprecated(since="0.18.0", removal="0.19.0")
def transfer_signal_axes(new_signal, old_signal):
    """Transfers signal axis calibrations from an old signal to a new
    signal produced from it by a method or a generator.

    Parameters
    ----------
    new_signal : Signal
        The product signal with undefined signal axes.
    old_signal : Signal
        The parent signal with calibrated signal axes.

    Returns
    -------
    new_signal : Signal
        The new signal with calibrated signal axes.
    """

    for i in range(old_signal.axes_manager.signal_dimension):
        ax_new = new_signal.axes_manager.signal_axes[i]
        ax_old = old_signal.axes_manager.signal_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


@deprecated(since="0.18.0", removal="0.19.0")
def transfer_navigation_axes(new_signal, old_signal):
    """Transfers navigation axis calibrations from an old signal to a new
    signal produced from it by a method or a generator.

    Parameters
    ----------
    new_signal : Signal
        The product signal with undefined navigation axes.
    old_signal : Signal
        The parent signal with calibrated navigation axes.

    Returns
    -------
    new_signal : Signal
        The new signal with calibrated navigation axes.
    """
    new_signal.transpose(
        len(new_signal.data.shape) - old_signal.axes_manager.navigation_dimension
    )

    for i in range(
        min(
            new_signal.axes_manager.navigation_dimension,
            old_signal.axes_manager.navigation_dimension,
        )
    ):
        ax_new = new_signal.axes_manager.navigation_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


@deprecated(since="0.18.0", removal="0.19.0")
def to_hyperspy_index(index, signal_axis):
    if isinstance(index, int):
        return index
    else:
        return signal_axis.value2index(index)


@deprecated(since="0.18.0", removal="0.19.0")
def transfer_navigation_axes_to_signal_axes(new_signal, old_signal):
    """Transfers navigation axis calibrations from an old signal to the signal
    axes of a new signal produced from it by a method or a generator.

    Used from methods that generate a signal with a single value at each
    navigation position.

    Parameters
    ----------
    new_signal : Signal
        The product signal with undefined navigation axes.
    old_signal : Signal
        The parent signal with calibrated navigation axes.

    Returns
    -------
    new_signal : Signal
        The new signal with calibrated signal axes.
    """
    for i in range(
        min(
            new_signal.axes_manager.signal_dimension,
            old_signal.axes_manager.navigation_dimension,
        )
    ):
        ax_new = new_signal.axes_manager.signal_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


def compute_markers(markers):
    """Compute the markers for a list of markers quickly by merging all of the dask arrays

    Parameters
    ----------
    markers : list
        List of markers.

    Returns
    -------
    markers : list
        List of markers now computed
    """
    to_compute = []
    for m in markers:
        for k, i in m.kwargs.items():
            if isinstance(i, da.Array):
                to_compute.append(i)
    computed = da.compute(to_compute)
    ind = 0
    for m in markers:
        for k, i in m.kwargs.items():
            if isinstance(i, da.Array):
                m.kwargs[k] = computed[0][ind]
                ind += 1
    return markers
