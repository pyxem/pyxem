"""
Mulit Phase Orientation Mapping
===============================
You can also calculate the orientation of the grains for multiple phases using the
:meth:`pyxem.signals.PolarSignal2D.get_orientation` method. This requires that you
simulate the entire S2 space for the phase and then compare to the simulated diffraction.
"""

import pyxem as pxm
from pyxem.data import fe_multi_phase_grains, fe_bcc_phase, fe_fcc_phase
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.quaternion import Rotation
from orix.sampling import get_sample_reduced_fundamental

mulit_phase = fe_multi_phase_grains()

# %%
# First we center the diffraction patterns and get a polar signal
# Increasing the number of npt_azim with give better polar sampling
# but will take longer to compute the orientation map
# The mean=True argument will return the mean pixel value in each bin rather than the sum
# this makes the high k values more visible

mulit_phase.calibration.center = None
polar_multi = mulit_phase.get_azimuthal_integral2d(
    npt=100, npt_azim=360, inplace=False, mean=True
)
polar_multi.plot()

# %%

# Now we can get make a simulation. In this case we want to set a minimum_intensity which removes the low intensity reflections.
# we also sample the S2 space using the :func`orix.sampling.get_sample_reduced_fundamental`
# We have two phases here so we can make a simulation object with both of the phases.
bcc = fe_bcc_phase()
fcc = fe_fcc_phase()

generator = SimulationGenerator(200, minimum_intensity=0.05)
rotations_bcc = get_sample_reduced_fundamental(
    resolution=1, point_group=bcc.point_group
)
rotations_fcc = get_sample_reduced_fundamental(
    resolution=1, point_group=fcc.point_group
)

sim = generator.calculate_diffraction2d(
    [bcc, fcc],
    rotation=[rotations_bcc, rotations_fcc],
    max_excitation_error=0.1,
    reciprocal_radius=2,
)
orientation_map = polar_multi.get_orientation(sim)

# mulit_phase.add_marker(orientation_map.to_single_phase_markers(annotate=True, text_color="w"))

# %%
# sphinx_gallery_thumbnail_number = 3
