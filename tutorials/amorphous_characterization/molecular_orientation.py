"""
# Orientation mapping for Molecular Glasses

This notebook looks into doing orientation mapping for polymers using hyperspy and pyxem.
This is a similar work flow for producing figures similar to those in the paper:

```
Using 4D STEM to Probe Mesoscale Order in Molecular Glass Films Prepared by Physical Vapor Deposition
Debaditya Chatterjee, Shuoyuan Huang, Kaichen Gu, Jianzhu Ju, Junguang Yu, Harald Bock, Lian Yu, M. D. Ediger, and Paul M. Voyles
Nano Letters 2023 23 (5), 2009-2015
DOI: 10.1021/acs.nanolett.3c00197
```

In this paper disk like structures in the glass are oriented in domains in the molecular glass.
 These domains result from Pi-Pi like stacking and the orientation of the structure can be measured by 4-D STEM.

Here we will go through the processing in pyxem/ hyperspy to create a figure similar to the image below which comes from the above paper.

<center><img src="data/12/ExampleImage.jpeg" style="height:400px"></center>

This is also a good example of how to develop custom workflows in pyxem.  This might eventaully be added as a supported feature to pyxem/hyperspy
using the `Model` class upstream in hyperspy but this requires that parallel processing in `hyperspy` when fitting signals is improved.

There are a couple of really cool things to focus on. Specifically this make heavy use of the `map` function in
 order to make these workflows both parallel and operate out of memory. This notebook is also designed to be easy
   to modify in the case that you have a different function that you want to fit!

The raw data used in section1 can be found at the link below:

https://app.globus.org/file-manager?origin_id=82f1b5c6-6e9b-11e5-ba47-22000b92c6ec&origin_path=%2Fmdf_open%2Fchatterjee_phenester_orientation_v1.3%2FFig2%2Fhttps://app.globus.org/file-manager?origin_id=82f1b5c6-6e9b-11e5-ba47-22000b92c6ec&origin_path=%2Fmdf_open%2Fchatterjee_phenester_orientation_v1.3%2FFig2%2F

"""

# %%

import hyperspy.api as hs
import pyxem as pxm
from pyxem.utils.ransac_ellipse_tools import determine_ellipse, _get_max_positions
import numpy as np
from skimage.morphology import disk, dilation

# %%
# Loading the Data
# ----------------
# We can acess the data using pyxem's data module.

s = pxm.data.organic_semiconductor(allow_download=True, lazy=True)


# %%
# Removing Ellipticity and Polar Unwrapping
# -----------------------------------------
# Often times you will have some ellipticity in a diffraction pattern or you
# might not know the exact center.

# In pyxem we have a method `determine_ellipse` which can be used to find some ellipse.
# This is useful for patterns where you don't have a zero beam to find the beam shift.  It is a
# pretty simple function, it just finds the max points in the diffraction pattern and
# uses those to define an ellipse.

# Let's make a mask for a beam stop and the zero beam.  In this case we can just
# set a high and a low threshold. We can also use a dilation to expand the mask.


mean_dp = s.mean()
mean_dp.compute()
mask = np.logical_not((mean_dp > 0.001) * (mean_dp < 0.8))
mask.data = dilation(mask.data, disk(14))

# %%
mask.plot()
# %%
# Now we can plot the "max" positions found in the diffraction pattern which we will use
# to fit an ellipse.
import matplotlib.pyplot as plt

pos = _get_max_positions(mean_dp, mask=mask.data, num_points=900)
plt.figure()
plt.imshow(mean_dp.data)
plt.scatter(pos[:, 1], pos[:, 0])
# %%

center, affine = determine_ellipse(mean_dp, mask=mask.data, num_points=1000)

# %%
# Pyxem has a method to unwrap the diffraction pattern into polar coordinates but it
# requires that you first calibrate the diffraction pattern.  The bare minimum is to
# set the center.  You can also set an affine transformation matrix to correct for
# ellipticity.
#
# We can test this by setting the center and affine on the mean diffraction pattern

mean_dp.set_signal_type("electron_diffraction")
mean_dp.calibration.center = center[::-1]  # note the reverse order of x and y
mean_dp.calibration.affine = affine

mean_dp.get_azimuthal_integral2d(npt=100, radial_range=(0.2, 0.75), mean=True).plot(
    vmax=0.4
)
# %%
# Now we can use the same calibration on the entire dataset and then unwrap the
# diffraction patterns into polar coordinates.

s.set_signal_type("electron_diffraction")
s.calibration.center = center[::-1]
s.calibration.affine = affine

s_polar = s.get_azimuthal_integral2d(npt=100, radial_range=(0.2, 0.75), mean=True)

# %%

pss = s_polar.isig[:, 0.25:0.35].sum(
    axis=-1
)  # Radial k-summed azimuthal intensity profiles
# %%

pss.compute()  # We can compute this because it is smaller now that it is 1D
pss.save(
    "data/PolarSum.zspy", overwrite=False
)  # Saving the data for use later (we are going to use some precomputed stuff which is a little larger)

# %%
# Processing the Polar Data
# -------------------------
# In this case all we really care about is the angle that the two arcs are located at from 0 to 180 degrees.
# The radial spectra have a fair bit of noise so we should think about filtering the data. In this case we can
# smooth the data before fitting the two arcs.  Using a larger sigma smooths the data more at the cost of losing small features in the dataset.


# Custom Smoothing Function
# -------------------------
# These are just some custom functions for filtering when there is a zero beam.
# It just ignores the zero beam when guassian filtering so that
# intensity doesn't bleed into the masked region

# Helper Functions (can ignore for the most part)
from scipy.ndimage import gaussian_filter1d
import numpy as np
from pyxem.utils.signal import to_hyperspy_index


def mask_gaussian1D(
    data,
    sigma,
    masked_region=None,
):
    """Gaussian smooth the data with a masked region which is ignored.

    Parameters
    ----------
    data: array-like
        A 1D array to be filtered
    sigma: float
        The sigma used to filter the data
    masked_region: tuple or None
        The region of the data to be ignored
    """
    if masked_region is not None:
        data_smooth = np.zeros(data.shape)
        data_smooth[0 : masked_region[0]] = gaussian_filter1d(
            data[0 : masked_region[0]], sigma
        )
        data_smooth[masked_region[1] :] = gaussian_filter1d(
            data[masked_region[1] :], sigma
        )
    else:
        data_smooth = gaussian_filter1d(data, sig)
    return data_smooth


def smooth_signal(signal, sigma, masked_region=None, **kwargs):
    """
    Helper function to smooth a signal.  The masked_region will use real units if the
    values are floats and pixel units if an int is passed
    """
    if masked_region is not None:
        masked_region = [
            to_hyperspy_index(m, signal.axes_manager.signal_axes[0])
            for m in masked_region
        ]
    return signal.map(
        mask_gaussian1D, sigma=sigma, masked_region=masked_region, **kwargs
    )


smoothed = smooth_signal(pss, sigma=5, masked_region=(-0.2, 0.1), inplace=False)

smoothed.plot()


# %%
# Fitting the Peaks
