"""
Setting Scales and Metadata
===========================

For the most part Rosettasciio will automatically get the scales and the metadata
from the file. Sometimes, however, you may want to set the scales and metadata yourself.
"""

from pyxem.data import simulated_strain

strained_signal = simulated_strain(
    navigation_shape=(32, 32),
    signal_shape=(512, 512),
    disk_radius=20,
    num_electrons=1e5,
    strain_matrix=None,
    lazy=True,
)


# %%
# Setting the scales
# -------------------
# The scales can be set using the ``set`` method. If you pass in a list it will
# automatically set the scales for each dimension. If you pass in a single value
# it will set that property for all dimensions.


strained_signal.axes_manager.navigation_axes.set(
    scale=0.1, units="nm", name=["x", "y"], offset=[2, 3]  # offset in nm,
)

strained_signal.axes_manager.signal_axes.set(
    scale=0.1, units="nm^-1", name=["kx", "ky"], offset=[2, 3]  # offset in nm
)

# %%
# Setting the Detector Gain
# -------------------------
# In most cases data is in units of detector counts or electrons.
# Setting the ``detector_gain`` property will set your plot to be
# in units of electrons.

strained_signal.calibration.detector_gain = 1

# %%
# Setting Other Metadata
# ----------------------
# Pyxem also has a number of other metadata properties that can be set using the
# calibration object for convenience. For example:

strained_signal.calibration.beam_energy = 200  # in keV
strained_signal.calibration.physical_pixel_size = (
    15e-6  # in meters, this is the physical pixel size of the detector
)
strained_signal.center = (
    None  # Set the center to be the center of the signal (changes offsets)
)
strained_signal.center = (
    256,
    256,
)  # Set the center to be at (256, 256) in pixel coordinates
strained_signal.calibration.convergence_angle = 1.5  # in mrad
