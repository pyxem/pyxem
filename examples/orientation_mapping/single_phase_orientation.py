"""
Single Phase Orientation Mapping
================================
You can also calculate the orientation of the grains in a single phase sample using the
:meth:`pyxem.signals.PolarSignal2D.get_orientation` method. This requires that you
simulate the entire S2 space for the phase and then compare to the simulated diffraction.
"""

from pyxem.data import si_phase, si_grains
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.sampling import get_sample_reduced_fundamental

simulated_si = si_grains()

# %%
# First we center the diffraction patterns and get a polar signal
# Increasing the number of npt_azim with give better polar sampling
# but will take longer to compute the orientation map
# The mean=True argument will return the mean pixel value in each bin rather than the sum
# this makes the high k values more visible

simulated_si.calibration.center = None
polar_si = simulated_si.get_azimuthal_integral2d(
    npt=100, npt_azim=360, inplace=False, mean=True
)
polar_si.plot()

# %%

# Now we can get make a simulation. In this case we want to set a minimum_intensity which removes the low intensity reflections.
# we also sample the S2 space using the :func`orix.sampling.get_sample_reduced_fundamental`
phase = si_phase()
generator = SimulationGenerator(200, minimum_intensity=0.05)
rotations = get_sample_reduced_fundamental(resolution=1, point_group=phase.point_group)
sim = generator.calculate_diffraction2d(
    phase, rotation=rotations, max_excitation_error=0.1, reciprocal_radius=2
)
orientation_map = polar_si.get_orientation(sim)
navigator = orientation_map.to_navigator()

simulated_si.plot(navigator=navigator, vmax="99th")
simulated_si.add_marker(
    orientation_map.to_single_phase_markers(annotate=True, text_color="w")
)

# %%
# sphinx_gallery_thumbnail_number = 3
