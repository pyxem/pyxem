"""
Centering the Zero Beam
=======================
"""

# %%
# Making a Dummy Dataset with a Zero Beam that systematically deviates from the center
import hyperspy.api as hs
import pyxem as pxm

s = pxm.data.tilt_boundary_data(correct_pivot_point=False)

# %%
# Getting the Position of the Zero beam
# -------------------------------------
# The zero beam position can be obtained using the :meth:`get_direct_beam_position` method.

s_shifts = s.get_direct_beam_position(method="blur", sigma=5, half_square_width=20)
s_shifts.plot()

# Getting a Linear Plane
# ---------------------
# In many instances the zero beam position will vary systematically with the scan position.
# This can be corrected by fitting a linear plane to the zero beam position using the
# :meth:`get_linear_plane` method.
s_linear_plane = s_shifts.get_linear_plane()
s_linear_plane.plot()

# %%
# Centering the Zero Beam
# -----------------------
# The zero beam can be centered using the :meth:`center_direct_beam` method.
# Then we sum all the diffraction patterns for the both the centered beam,
# and the non-centered one, to compare them.

s_centered = s.center_direct_beam(shifts=s_linear_plane, inplace=False)
s_pacbed_centered = s_centered.sum()
s_pacbed = s.sum()

hs.plot.plot_images([s_pacbed, s_pacbed_centered], label=["Original", "Centered"])

# %%
# Centering the Zero Beam with constant deflection magnitude
# -----------------------
# In the presence of electromagnetic fields in the entire sample area, 
# the plane fitting can fail However, if there are several domains expected 
# to have equal deflection magnitude, we can try to fit a plane to this by 
# minimizing the magnitude variance.

def direct_beam_dataset_with_constant_shift_magnitude():

    import numpy as np
    bright_field_disk = np.zeros((128,128),dtype=np.int16)
    bright_field_disk[np.sum((np.mgrid[:128,:128]-64)**2,axis=0) < 10**2 ] = 500

    probes = np.zeros((20,20),dtype=int)
    probes = bright_field_disk[np.newaxis][probes]
    probes = pxm.signals.Diffraction2D(probes)


    p = [0.5] * 6  # Plane parameters
    x, y = np.meshgrid(np.arange(20), np.arange(20))
    base_plane_x = p[0] * x + p[1] * y + p[2]
    base_plane_y = p[3] * x + p[4] * y + p[5]

    base_plane = np.stack((base_plane_x, base_plane_y)).T
    data = base_plane.copy()

    shifts = np.zeros_like(data)
    shifts[:10, 10:] = (10, 10)
    shifts[:10, :10] = (10, -10)
    shifts[10:, 10:] = (-10, 10)
    shifts[10:, :10] = (-10, -10)
    data += shifts
    data = pxm.signals.BeamShift(data)
    probes.center_direct_beam(shifts=-data)
    return probes

probes = direct_beam_dataset_with_constant_shift_magnitude()

s_shifts = probes.get_direct_beam_position(method="center_of_mass")

# %%
# We call `get_linear_plane` with `constrain_magnitude_variance=True`. 
s_linear_plane = s_shifts.get_linear_plane(constrain_magnitude_variance=True)
s_shifts.data -= s_linear_plane.data

probes.center_direct_beam(shifts=s_linear_plane)
probes.plot()

# %%
# The found electromagnetic domains can be visualized like such.
s_shifts.get_magnitude_phase_signal().plot()
# %%
