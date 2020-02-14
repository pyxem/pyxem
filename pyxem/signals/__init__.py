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


def push_metadata_through(dummy, *args, **kwargs):
    """
    This function pushes loaded metadata through to pyxem objects, it is to be used for one
    purpose, see the __init__ of ElectronDiffraction2D for an example.

    Parameters
    ----------
    dummy :
        This will always be the "self" of the object to be initialised

    args : list
        This will always be the "args" of the object to be initialised

    kwargs : dict
        This will always be the "args" of the object to be initialised

    Returns
    -------
    dummy,args,kwargs :
        The input variables, adjusted correctly
    """
    try:
        meta_dict = args[0].metadata.as_dictionary()
        kwargs.update({'metadata': meta_dict})
    except AttributeError:
        pass  # this is because a numpy array has been passed
    except IndexError:
        pass  # this means that map continues to work.

    return dummy, args, kwargs


def select_method_from_method_dict(method, method_dict, **kwargs):
    """
    Streamlines the selection of utils to be mapped in class methods

    Parameters
    ----------
    method : str
        The key to method_dict for the chosen method

    method_dict : dict
        dictionary with strings as keys and functions as values

    kwargs : dict
        Parameters for the method, if empty help is return

    Returns
    -------
    method_function :
        The utility function that corresponds the given method string, unless
        kwargs is empty, in which case the help for the utility function is returned.
    """

    if method not in method_dict:
        raise NotImplementedError("The method `{}` is not implemented. "
                                  "See documentation for available "
                                  "implementations.".format(method))
    elif not kwargs:
        help(method_dict[method])

    return method_dict[method]


def transfer_signal_axes(new_signal, old_signal):
    """ Transfers signal axis calibrations from an old signal to a new
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


def transfer_navigation_axes(new_signal, old_signal):
    """ Transfers navigation axis calibrations from an old signal to a new
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
    new_signal.axes_manager.set_signal_dimension(
        len(new_signal.data.shape) - old_signal.axes_manager.navigation_dimension)

    for i in range(min(new_signal.axes_manager.navigation_dimension,
                       old_signal.axes_manager.navigation_dimension)):
        ax_new = new_signal.axes_manager.navigation_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


def transfer_navigation_axes_to_signal_axes(new_signal, old_signal):
    """ Transfers navigation axis calibrations from an old signal to the signal
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
    for i in range(min(new_signal.axes_manager.signal_dimension,
                       old_signal.axes_manager.navigation_dimension)):
        ax_new = new_signal.axes_manager.signal_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal
