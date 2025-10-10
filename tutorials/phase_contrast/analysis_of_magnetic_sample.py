"""
# Analysing magnetic materials using STEM-DPC

This notebook shows how to use the `pyXem` library to analyse 4-D scanning transmission electron microscopy (STEM) data, specifical magnetic materials using differential phase contrast (DPC). For more information about this imaging method, see the Wikipedia article on Scanning Transmission Electron Microscopy, which has a subsection on DPC: https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy#Differential_phase_contrast

The data we'll be looking at there is from the paper **Strain Anisotropy and Magnetic Domains in Embedded Nanomagnets**, and is STEM data recorded on a Merlin fast pixelated electron detector system, where the objective lens has been turned off.
This allows for magnetic information to be extracted, by carefully mapping the beam shifts.

More documentation about pyXem is found at https://pyxem.readthedocs.io/

Journal article:
* **Strain Anisotropy and Magnetic Domains in Embedded Nanomagnets**
* Nord, M., Semisalova, A., Kákay, A., Hlawacek, G., MacLaren, I., Liersch, V., Volkov, O. M., Makarov, D., Paterson, G. W., Potzger, K., Lindner, J., Fassbender, J., McGrouther, D., Bali, R.
* Small 2019, 15, 1904738. https://doi.org/10.1002/smll.201904738

The full dataset and scripts used in analysing this data is found at Zenodo: https://zenodo.org/record/3466591

This notebook has been modified to use a cropped version of the data with only the infromation about the zero beam.

Refer to the link above to look at the entire dataset.
"""

# %%
import hyperspy.api as hs
import pyxem as pxm

# Load the data
# The data is available at https://zenodo.org/record/3466591
# Or using pyxem's data module

s = pxm.data.feal_stripes(allow_download=True, lazy=True)  # lazy loading using dask

s

# %%
# plot the data to visualise it

s.plot()


# %%

# This dataset is chunked into blocks of (16 x 16 x 16 x 16) hyperpixels (x, y, kx, ky)
# This means that we can create virtual images quite easily and quickly.
# For example:
# %%

s_nav = s.isig[32, 35].T
s_nav.plot()
# %%
# We can use this as our navigation image:

# %%
s.plot(navigator=s_nav)

# Many Times the shifts are quite small so one thing we can do is transpose
# the data and then interactively shift around to visualize the shift.

# Try using Shift + Left Mouse Button to jump to the Edge of the Zero beam
# in the Transposed Image:

s_diff_nav = s.inav[10:20, 10:20].mean(axis=(0, 1))
s_diff_nav.plot()

# %%

s_transpose = s.transpose()

s_transpose.plot(navigator=s_diff_nav)

# %%
# This visualizes the shift of the beam, which is caused by the beam passing through the ferromagnetic domains in the material.
# However, it is not very quantitative. So lets try to extract the beam shifts using center of mass.

# Extracting the Beam Shifts using Center of Mass
# -----------------------------------------------

com = s.get_direct_beam_position(
    method="center_of_mass", half_square_width=40  # in pixels
)

com
# %%
# This returns a `BeamShift1D` class, which will be explored more later. What we need to know is that
# is it basically a HyperSpy `Signal1D` class, where at each navigation pixel we have a 2D vector
# (kx, ky) of the beam shift.

# Plotting it shows the seperated x, and y components of the beam shift.
# You first need to compute it before you can plot it.

com.compute()
com.plot()

# %%
# Other Contrast:
# ---------------
# However, we're also getting contrast from the other effects, such as structural effects.
#  Since the sample is nanocrystalline, some of the grains will be close to some zone axis,
#  giving heavy Bragg scattering. While the Bragg spots themselves won't be visible at such low
# scattering angles as we have in `s_crop`, it will still change the intensity distribution _within_
# the direct beam. Essentially, the direct beam becomes non-uniform, which will have an effect similarly
#  to beam shift.
#
# One way of reducing this is by using thresholding and masking. However, we first need to find reasonable
#  values for these.
# For this, we use `threshold_and_mask` on a subset of the dataset.

s_threshold_mask = s.threshold_and_mask(
    threshold=1,
    mask=(64, 64, 46),  # x  # y  # radius
)

s_threshold_mask.plot(navigator=s_nav)


# `threshold_and_mask` is a useful way to preprocess the data before extracting the beam shifts.
# It works by getting the mean of the intensity inside the masked area, times the threshold.
#  Then, any pixel lower or equal than that value is set to zero, while any value above that
#  value is set to one. Ideally, this should remove the influence of other diffraction effects,
# and non-uniform direct beam.
# %%
com_threshold = s_threshold_mask.get_direct_beam_position(
    method="center_of_mass", half_square_width=40  # in pixels
)

com_threshold.compute()
com_threshold.plot()

# %%

# Correcting the d-scan
# ---------------------
# With the beam shift extracted, we will remove the effects of impure beam shift (d-scan).
# This is due to various instrument misalignments, and leads to a change in beam position in
# the probe plane becoming a shift of the beam in the detector plane.
# Luckily, in most situations, the d-scan is linear across the dataset, meaning it can be removed
# using a simple plane subtraction.


plane = com_threshold.get_linear_plane()
plane.plot()


# %%
# Visualising the Beam Shifts
# ---------------------------
# Now we can visualize the signal as a magnitude and direction maps: `get_color_signal`,`get_magnitude_phase_signal`,
#  `get_magnitude_signal`
# and `get_color_image_with_indicator`.

# The two former returns a HyperSpy signal, while the latter interfaces directly with the matplotlib
# backend making it more customizable.
com_threshold_corrected = com_threshold - plane
com_corrected = com - plane
com_threshold_corrected.get_color_signal().plot()


# %%
com_threshold_corrected.get_magnitude_phase_signal().plot()

# %%
com_corrected.get_magnitude_phase_signal().plot()

# %%
