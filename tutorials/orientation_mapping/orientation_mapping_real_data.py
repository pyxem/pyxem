"""
# Orientation Mapping of Polycrystalline Silicon

Orientation mapping in 4D-STEM determines the crystallographic orientation at
every scan position across a polycrystalline or multi-grain sample.  Unlike
EBSD (which only samples the surface), 4D-STEM captures the full diffraction
geometry and can resolve grains in bulk and thin specimens alike.

## How it works: template matching in polar space

The core idea is **template matching**:

1. **Simulate** diffraction patterns for every orientation in the crystal's
   reduced fundamental zone (the irreducible wedge of orientation space for
   that point-group symmetry).
2. **Convert** both the experimental data and the library to polar coordinates
   ``(|k|, φ)`` — this makes the comparison invariant to in-plane rotation
   and keeps memory use manageable.
3. **Cross-correlate** each experimental polar pattern against every simulated
   template and record the best-matching orientation.

The result is an ``OrientationMap`` — a HyperSpy signal where each navigation
pixel stores the best-matching rotation, correlation score, and phase index.

This tutorial uses a real 4D-STEM dataset of Si thin film recorded at 200 kV
with multiple overlapping grains at different orientations.

Dataset: https://zenodo.org/records/11284654
Reference: :cite:`pyxemorientationmapping2022`
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pyxem as pxm
from pyxem.data import si_phase, sample_with_g
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.sampling import get_sample_reduced_fundamental

# %%
# Loading the Data
# ----------------
# ``sample_with_g`` is a calibrated 4D-STEM dataset of a multi-grain Si thin
# film.  The scan is 25 × 26 real-space positions, each recording a
# 256 × 256 pixel diffraction pattern at 200 kV.

s = sample_with_g(allow_download=True)
print(s)
print(s.axes_manager)

# %%
# Plot a single diffraction pattern from the centre of the scan.  You should
# see sharp Si diffraction spots with clear crystal symmetry.

s.inav[13, 12].plot(vmax="99th")

# %%
# Plot the mean diffraction pattern across all scan positions.  Because
# multiple grain orientations are averaged together the pattern looks like
# a powder ring, but individual spots are still visible in single patterns.

s.mean().plot(vmax="99th", cmap="magma")

# %%
# Pre-Processing: Centring the Direct Beam
# -----------------------------------------
# The centre of the diffraction pattern must coincide with the zero-beam
# (transmitted beam) position.  ``calibration.center = None`` auto-detects
# the centre by finding the brightest pixel and centering the axes offsets.
#
# Accurate centering is critical: a shift of even one pixel translates into
# an orientation error, because the cross-correlation treats the polar-space
# radius as the scattering vector magnitude.

s.calibration.center = None
print("Signal axes after centring:")
print(s.axes_manager)

# %%
# Azimuthal Integration to Polar Coordinates
# ------------------------------------------
# ``get_azimuthal_integral2d`` re-bins each diffraction pattern into polar
# coordinates ``(|k|, φ)``.  Key parameters:
#
# - ``npt``: number of radial bins.  More bins = finer |k| resolution,
#   but also more memory.  100 is a good default for orientation mapping.
# - ``npt_azim``: number of azimuthal bins.  360 gives 1° resolution.
#   Increasing this improves angular sensitivity at the cost of compute time.
# - ``mean=True``: return the mean pixel value per bin rather than the sum.
#   This equalises the weight of inner and outer rings, making high-|k|
#   reflections visible alongside the bright low-|k| spots.

polar = s.get_azimuthal_integral2d(
    npt=100,
    npt_azim=360,
    inplace=False,
    mean=True,
)
print(polar)

# %%
# Plot the mean polar pattern.  The x-axis is azimuthal angle (0–360°) and
# the y-axis is scattering vector magnitude |k|.  The bright horizontal bands
# correspond to the Si ring positions.

polar.mean().plot(cmap="magma", vmax="99th")

# %%
# Building the Simulation Library
# ---------------------------------
# We need a simulated diffraction pattern for every orientation in the reduced
# fundamental zone of Si (point group m-3m, cubic).
#
# **Sampling orientation space**
#
# ``get_sample_reduced_fundamental(resolution=1, point_group=...)`` returns a
# uniform grid of rotations covering the irreducible wedge.  ``resolution=1``
# means one rotation per degree — giving ~136 rotations for cubic symmetry.
# Finer grids (e.g. ``resolution=0.5``) give better accuracy but are slower.

phase = si_phase()
print(f"Phase: Silicon, space group: {phase.space_group.short_name}")
print(f"Point group: {phase.point_group.name}, a = {phase.structure.lattice.a:.4f} Å")

rotations = get_sample_reduced_fundamental(
    resolution=1,
    point_group=phase.point_group,
)
print(f"Number of orientations sampled: {rotations.size}")

# %%
# **Simulation parameters**
#
# ``SimulationGenerator`` uses the kinematical (single-scattering) approximation
# to compute structure factors and diffraction spot intensities.
#
# - ``accelerating_voltage=200``: beam energy in kV.  Affects the Ewald sphere
#   radius and therefore which reflections are excited.
# - ``minimum_intensity=0.05``: discard reflections weaker than 5% of the
#   strongest reflection.  This removes many forbidden and very weak spots that
#   would add noise to the template library without improving matching.
#
# ``calculate_diffraction2d`` parameters:
#
# - ``max_excitation_error=0.1``: the maximum deviation from the Bragg
#   condition (in reciprocal Å) at which a reflection is still included.
#   Larger values = more spots per pattern = better match for tilted grains.
#   0.1 Å⁻¹ is a good default for thin TEM samples.
# - ``reciprocal_radius=2``: only simulate reflections within 2 Å⁻¹ of the
#   origin (matching the detector range of the experimental data).
# - ``with_direct_beam=False``: **always** exclude the transmitted beam.
#   Including it would make the cross-correlation dominated by the bright
#   centre spot, which carries no orientation information.

generator = SimulationGenerator(
    accelerating_voltage=200,
    minimum_intensity=0.05,
)

sim = generator.calculate_diffraction2d(
    phase,
    rotation=rotations,
    max_excitation_error=0.1,
    reciprocal_radius=2,
    with_direct_beam=False,
)
print(f"Simulation library built for {rotations.size} orientations.")

# %%
# Gamma Correction
# -----------------
# Before matching, we apply a gamma correction to the polar signal:
#
# .. code-block:: python
#
#     polar = polar ** 0.5   # equivalent to gamma = 0.5
#
# **Why?**  In a raw diffraction pattern the central/low-|k| spots are orders
# of magnitude brighter than the outer reflections.  The cross-correlation
# would then be dominated by these bright spots, ignoring the rich angular
# information in the weaker high-|k| reflections.
#
# Raising the pattern to the power 0.5 (square root) compresses the dynamic
# range, giving the fainter spots more influence without discarding them.
# You can tune this exponent: lower values weight weak spots more heavily,
# higher values (closer to 1) preserve the natural intensity hierarchy.
# For noisy data, background subtraction before gamma correction often helps.

polar_gamma = polar**0.5

# Visualise the effect of gamma correction on a single pattern.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(polar.inav[13, 12].data, aspect="auto", cmap="magma")
ax1.set_title("Raw polar pattern")
ax1.set_xlabel("Azimuthal angle (bins)")
ax1.set_ylabel("|k| (bins)")
ax2.imshow(polar_gamma.inav[13, 12].data, aspect="auto", cmap="magma")
ax2.set_title("After gamma correction (γ = 0.5)")
ax2.set_xlabel("Azimuthal angle (bins)")
plt.tight_layout()

# %%
# Running the Orientation Mapping
# --------------------------------
# ``get_orientation`` cross-correlates every scan-position polar pattern
# against every template in the library and records the best match.
#
# The result is an ``OrientationMap`` where each navigation pixel stores:
# - the best-matching rotation (as an orix ``Rotation``)
# - the normalised cross-correlation score (0–1)
# - the phase index (always 0 for single-phase data)
#
# For a 25 × 26 scan and ~136 templates this typically takes a few seconds
# on a modern CPU.  For larger datasets (e.g. 256 × 256 scan) GPU acceleration
# via CuPy can speed this up by 10–100×.

orientation_map = polar_gamma.get_orientation(sim)
print(orientation_map)

# %%
# Visualising the Orientation Map
# --------------------------------
# ``plot_over_signal`` overlays an Inverse Pole Figure (IPF) colour map on the
# experimental virtual bright-field image.
#
# In an IPF map:
# - Each pixel is coloured according to the crystal direction aligned with the
#   beam (z-axis by default).
# - The colour scheme uses the standard cubic IPF triangle: red = [001],
#   green = [101], blue = [111].
# - Pixels with similar colours share a common zone axis.
# - Abrupt colour changes indicate grain boundaries.
#
# ``vmax="96th"`` clips the intensity at the 96th percentile to prevent a few
# very bright pixels from washing out the contrast in the real-space image.

orientation_map.plot_over_signal(s, vmax="96th")

# %%
# Correlation Score Map
# ----------------------
# The correlation score at each pixel indicates how well the best-matching
# template reproduces the experimental pattern.  Low scores may indicate:
# - grain boundaries (two overlapping grains, neither matches well alone)
# - amorphous regions or surface contamination
# - patterns too close to a zone-axis (kinematical approximation breaks down
#   under strongly channelled conditions)

orientation_map.plot_over_signal(s, vmax="96th", add_ipf_correlation_heatmap=True)

# %%
# Exporting to a Crystal Map
# ---------------------------
# ``to_crystal_map()`` converts the ``OrientationMap`` to an orix
# ``CrystalMap`` object, which is the standard container for orientation data
# in the pyxem/orix ecosystem.  From the crystal map you can:
#
# - compute misorientations between neighbouring pixels
# - apply smoothing or grain segmentation algorithms
# - export to .ang or .ctf format for comparison with EBSD

cmap = orientation_map.to_crystal_map()
cmap.plot(
    "phase_name",
    legend=True,
)

# %%
# Tips for Better Results
# ------------------------
# **Resolution**: ``resolution=1`` is a coarse starting point.  For accurate
# misorientations, use ``resolution=0.5`` or finer.  This squares the number
# of templates and proportionally increases compute time.
#
# **Reciprocal radius**: set to match the actual calibrated range of your
# detector.  Too large wastes compute time on empty space; too small misses
# high-|k| information.
#
# **Noise reduction**: averaging each pattern with its neighbours before
# template matching (``s.map(lambda p: gaussian_filter(p, 1))``) often
# improves the score in low-dose datasets.
#
# **Multi-phase datasets**: pass a list of phases and rotations to
# ``calculate_diffraction2d`` to simultaneously index multiple phases.  See
# the multi-phase example in the examples gallery for details.

# %%
# sphinx_gallery_thumbnail_number = 5
