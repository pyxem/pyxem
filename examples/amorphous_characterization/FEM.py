"""
Fluctuation Electron Microscopy
===============================
This example shows you how to perform a fluctuation electron microscopy (FEM) analysis.

In this example we will focus on calculating :math:`V_\\Omega (k)` which is defined as,

Equation 1:

.. math::

    V_\\Omega (k) = \\frac{\\langle I{^2}(k) \\rangle{_r} - \\langle I(k)\\rangle_r^2}{\\langle I(k)\\rangle_r^2 } - \\frac{G}{\\langle I(k)\\rangle_r}

where :math:`I(k)` is the diffracted electron intensity averaged over the polar angle at constant scattering
vector magnitude k, :math:`<>_r` indicates averaging over probe positions, and G is the gain of the electron camera
in counts per electron for counted or data from hybrid pixel detectors this value is 1 otherwise it will be some
mean value. The first term is the definition of the variance, and the second term is a correction to
the variance for Poisson noise in the data.

(There are several different possible variance signals. Here, we use the notation from
Daulton, et al. Ultramicroscopy 110, 1279–1289 (2010), DOI: 10.1016/j.ultramic.2010.05.010.)
"""

import pyxem as pxm
from pyxem.utils import determine_ellipse
import numpy as np
import hyperspy.api as hs
from pyxem.utils._pixelated_stem_tools import _copy_axes_object_metadata

s = pxm.data.zrcual_1(allow_download=True, signal_type="electron_diffraction")
s.plot()

# %%
# Using Manual Ellipse Determination
# ----------------------------------
# Sometimes it is useful to force the ellipse to fit certain points.  For example, here we
# can force the ellipse to fit the first ring by masking the zero beam.

summed = s.sum()

center, affine, params, pos = pxm.utils.ransac_ellipse_tools.determine_ellipse(
    summed,
    return_params=True,
    num_points=500,
    use_ransac=False,
)
el, in_points = pxm.utils.ransac_ellipse_tools.ellipse_to_markers(
    ellipse_array=params,
    points=pos,
)

# we don't account for scales/offsets yet
summed.axes_manager.signal_axes[0].scale = 1
summed.axes_manager.signal_axes[1].scale = 1
summed.axes_manager.signal_axes[1].offset = 0
summed.axes_manager.signal_axes[0].offset = 0


# %%
# Checking
# --------
# Let's check to make sure that things are behaving.  We can first plot the ellipse over
# the data and then take the azimuthal integral/sum.
# That should end up a nice straight line

summed.plot()
summed.add_marker(in_points, plot_marker=True)
summed.add_marker(el, plot_marker=True)


s.calibration.center = center[::-1]  # reverse the center.
s.calibration.affine = affine
az = s.get_azimuthal_integral2d(npt=100).sum().isig[:, 2.0:8.0]

az.sum().plot()

# %%
# Getting the Variance
# --------------------
# The :meth:`~.signals.diffraction2d.get_variance` function will calculate the variance using the affine correction
# and the center as described above. Restricting the radial range is also nice to remove
# the effects of the high intensity at the top end. Adding a mask can also be helpful for
# reducing the effects of a beam stop. The ``gain`` parameter is number of detector units for 1 electron.
# It's used for the Poisson noise correction. If the data are already calibrated in units of electron counts,
# use a gain of 1.

mask = summed < 10000
mask.plot()
variance = s.get_variance(npt=50, gain=4.2, radial_range=(3.0, 5.75), mask=mask)
variance.axes_manager[0].units = "$nm^{-1}$"
variance.plot()

# %%
# Getting a (Good) Variance
# -------------------------
# If the TEM sample varies in thickness by more than a few nm over the entire dataset,
# thickness-related differences in the diffracted intensity  will dominate structure-related
# differences and therefore dominate. This effect can be avoided by using the HAADF signal or
# the high-angle scattering within the diffraction pattern to determine the local sample thickness,
# grouping the diffraction patterns into bins of nearly constant thickness, and only computing :math:`V_\\Omega (k)`
# for data inside a single bin. The :math:`V_\\Omega (k)` from different thickness bins then can be averaged
# together. See Hwang and Voyles Microscopy and Microanalysis 17, 67–74 (2011), DOI: 10.1017/S1431927610094109
# and Li et al. Microscopy and Microanalysis 20, 1605–1618 (2014). DOI: 10.1017/s1431927614012756 for more details.

# We have already saved the simultaneously-acquired HAADF image along side the dataset. We can see this here...
haadf = s.metadata.HAADF.intensity

# %%
# The HAADF signal for an amorphous material is linear in the thickness for typical TEM sample thicknesses.
# To convert the HAADF digital counts to thickness (in e.g. nm), the slope and intercept of the linear
# relationship must be known. The intercept is the black level of the HAADF detector in digital counts,
# which in this case is 26,265. The slope must be calibrated for each experiment from a measurement of
# the HAADF intensity at a position of known sample thickness. In this case, the slope is 440.46 digital
# counts / nm of sample thickness.
#
# TEM sample thickness for amorphous materials can be measured independently either using electron energy
# loss spectroscopy (EELS), in which the inelastic mean free path of the material must be known, or using total
# elastic scattering, in which case the elastic mean free path of the material must be known. A reasonable model
# for the elastic mean free path for many inorganic materials may be found
# in Zhang et al. Ultramicroscopy 171, 89–95 (2016), DOI: 10.1016/j.ultramic.2016.09.005.

thickness = (haadf - 26265) / 440.46


def thickness_filter(signal, thickness, bins):
    masks = [
        np.logical_and(bins[i] < thickness, bins[i + 1] > thickness)
        for i in range(len(bins) - 1)
    ]
    filtered = [hs.signals.Signal2D(signal.data[m.data, :, :]) for m in masks]
    for f in filtered:
        f.set_signal_type("electron_diffraction")
        _copy_axes_object_metadata(
            signal.axes_manager.signal_axes[0], f.axes_manager.signal_axes[0]
        )
        _copy_axes_object_metadata(
            signal.axes_manager.signal_axes[1], f.axes_manager.signal_axes[1]
        )
        f.metadata.add_dictionary(signal.metadata.as_dictionary())
    return filtered, thickness


bins = np.linspace(
    np.min(thickness, axis=(0, 1)), np.max(thickness, axis=(0, 1)), num=2 + 1
)
filtered, thickness = thickness_filter(s, thickness, bins)
var = [
    f.get_variance(npt=50, gain=4.2, radial_range=(3.0, 5.7), mask=mask)
    for f in filtered
]

# Note that the y-axis is the variance here. Hyperspy just always labels this as "Intensity"
for v in var:
    v.axes_manager[0].units = "$nm^{-1}$"
hs.plot.plot_spectra(var, legend=["thickness<17.5nm", "thickness<18.5nm"])
