import numpy as np

from hyperspy.api import interactive


class CommonDiffraction:

    @staticmethod
    def _get_sum_signal(signal, out_signal_axes=None):
        out = signal.sum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2)))
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError("The length of 'out_signal_axes' can't be longer"
                             "than the navigation dimension of the signal.")
        return out.transpose(out_signal_axes)

    def plot_interactive_virtual_image(self, roi, out_signal_axes=None,
                                       **kwargs):
        """Plots an interactive virtual image formed with a specified and
        adjustable roi

        Parameters
        ----------
        roi : float
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes : None, iterable of int or string
            Specify which navigation axes to use as signal axes in the virtual
            image. If None, the two first navigation axis are used.
        **kwargs:
            Keyword arguments to be passed to the `plot` method of the virtual
            image.

        Examples
        --------
        .. code-block:: python

            >>> # For 1D diffraction signal, we can use a SpanROI
            >>> roi = hs.roi.SpanROI(left=1., right=2.)
            >>> rp.plot_interactive_virtual_image(roi)

        .. code-block:: python

            >>> # For 2D diffraction signal,we can use a CircleROI
            >>> roi = hs.roi.CircleROI(3, 3, 5)
            >>> rp.plot_interactive_virtual_image(roi)

        """
        # Plot signal when necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the roi
        sliced_signal = roi.interactive(self,
                                        axes=self.axes_manager.signal_axes)

        # Create an output signal for the virtual dark-field calculation.
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Virtual Dark Field"

        # Create the interactive signal
        interactive(sliced_signal.sum,
                    axis=sliced_signal.axes_manager.signal_axes,
                    event=roi.events.changed,
                    recompute_out_event=None,
                    out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    def get_virtual_image(self, roi, out_signal_axes=None):
        """Obtains a virtual image associated with a specified scattering range.

        Parameters
        ----------
        roi : :obj:`hyperspy.roi.BaseInteractiveROI`
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes : None, iterable of int or string
            Specify which navigation axes to use as signal axes in the virtual
            image. If None, the two first navigation axis are used.

        Returns
        -------
        dark_field_sum : :obj:`hyperspy.signals.Signal2D` or :obj:`hyperspy.signals.Signal1D`
            The virtual image signal associated with the specified scattering
            range.

        Examples
        --------
        .. code-block:: python

            rp.get_virtual_image(left=0.5, right=0.7)

        """
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = self._get_sum_signal(dark_field, out_signal_axes)
        dark_field_sum.metadata.General.title = f"Virtual Dark Field ({roi})"

        return dark_field_sum
