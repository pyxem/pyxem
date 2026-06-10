"""
Angular Correlations of Amorphous Materials
============================================

Amorphous materials lack long-range order, but they often have characteristic
**short-range** and **medium-range** order that can be probed with 4D-STEM.
Angular correlation analysis measures the statistical relationship between
the diffracted intensity at two different azimuthal angles at the same
scattering vector magnitude |k|.  Peaks in the angular power spectrum reveal
preferred bond angles — for example, five-fold or icosahedral symmetry in
metallic glasses.

This tutorial demonstrates the full workflow on a PdNiP metallic glass dataset:

1. Loading and calibrating the data
2. Azimuthal integration to polar coordinates
3. Computing the angular autocorrelation function
4. Computing the angular power spectrum
5. Identifying dominant symmetry orders

The dataset is available at https://zenodo.org/records/11284654

Reference: E. Bokeloh et al., *Phys. Rev. Lett.* 107, 145701 (2011).
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pyxem as pxm

# %%
# Loading the Data
# ----------------
# The PdNiP glass dataset is a 128 × 128 scan with 128 × 128 diffraction patterns.
# The data is already signal-typed as ElectronDiffraction2D and calibrated.

s = pxm.data.pdnip_glass(allow_download=True)
print(s)
print(s.axes_manager)

# %%
# For tutorial speed we operate on a 32×32 subset of the 128×128 scan.
# The angular statistics are qualitatively the same on this smaller region.

s = s.inav[:32, :32]
print("Subset:", s)

# %%
# Plot a mean diffraction pattern to see the characteristic broad amorphous rings.

s.mean().plot(vmax=2000, cmap="viridis")

# %%
# Calibration
# -----------
# The signal axes already have correct scale and units set.
# We just need to specify the beam centre. The axes have a small offset, so the
# centre in pixel units is:

cx = -s.axes_manager.signal_axes[1].offset / s.axes_manager.signal_axes[1].scale
cy = -s.axes_manager.signal_axes[0].offset / s.axes_manager.signal_axes[0].scale
print(f"Centre: ({cx:.2f}, {cy:.2f}) pixels")

s.calibration(center=(cx, cy))

# %%
# Azimuthal Integration to Polar Coordinates
# -------------------------------------------
# ``get_azimuthal_integral2d`` bins the diffraction intensity into (|k|, φ) bins,
# producing a ``PolarDiffraction2D`` signal.  The result preserves all the
# angular information that would be lost in a 1D radial profile.

# Use a radial range that covers the first two amorphous rings.
k_min_nm = 2.0  # nm⁻¹
k_max_nm = 12.0  # nm⁻¹

pol = s.get_azimuthal_integral2d(
    npt=60,  # radial bins
    npt_azim=180,  # azimuthal bins (2° per bin)
    radial_range=(k_min_nm, k_max_nm),
)
print(pol)

# %%
# Plot the mean polar pattern: the x-axis is azimuthal angle and the y-axis
# is the scattering vector magnitude.

pol.mean().plot(cmap="viridis")

# %%
# Angular Autocorrelation
# -----------------------
# The angular autocorrelation C(|k|, Δφ) measures how the intensity at angle φ
# correlates with the intensity at angle φ + Δφ, averaged over all φ.
# A peak at Δφ = 2π/n indicates n-fold symmetry.

cor = pol.get_angular_correlation(normalize=True)
print(cor)

# %%
# Plot the mean angular correlation: prominent peaks at characteristic Δφ values
# reveal the local bond-angle preferences.

cor.mean().plot(cmap="RdBu_r", vmin=-0.5, vmax=0.5)

# %%
# Angular Power Spectrum
# ----------------------
# The angular power spectrum decomposes the correlation function into its
# Fourier harmonics.  Each harmonic l corresponds to l-fold symmetry.
# Amorphous materials often show strong contributions at l = 5 (icosahedral)
# or l = 6 (face-centred cubic short-range order).

power = cor.get_angular_power()
print(power)

# %%
# Sum over all scan positions to get the dataset-averaged power spectrum.

power_mean = power.mean()
power_mean.plot(cmap="hot")

# %%
# Radial Dependence of Symmetry
# ------------------------------
# Different symmetry orders often dominate at different values of |k|.
# ``power_mean`` has display axes (harmonics | radial), so ``isig[l]`` selects
# harmonic l and returns a 1D radial profile.

import hyperspy.api as hs

# Extract individual harmonics (l = 2, 4, 5, 6, 10)
harmonics = [2, 4, 5, 6, 10]
spectra = [power_mean.isig[l] for l in harmonics]
labels = [f"l = {l}" for l in harmonics]

hs.plot.plot_spectra(spectra, legend=labels, style="cascade")
plt.xlabel("|k| (nm⁻¹)")
plt.ylabel("Power")
plt.title("Angular power spectrum by harmonic order")

# %%
# Spatial Map of Dominant Symmetry
# ----------------------------------
# Summing the power over all radial bins at each harmonic gives a
# 2D spatial map of that symmetry order.  We use the underlying numpy arrays
# directly for clean matplotlib control.
#
# ``power`` has numpy shape (ny, nx, n_radial, n_harmonics).
# Axis order: power.data[y, x, r, l].

n_harmonics_axis = power.data.shape[-1]
n_radial_axis = power.data.shape[-2]

fig, axs = plt.subplots(1, len(harmonics), figsize=(14, 3))
for ax, l in zip(axs, harmonics):
    if l < n_harmonics_axis:
        spatial = power.data[..., l].sum(axis=-1)  # sum over radial
        ax.imshow(spatial, cmap="hot", origin="upper")
    ax.set_title(f"l = {l}")
    ax.axis("off")
plt.tight_layout()

# sphinx_gallery_thumbnail_number = 4
