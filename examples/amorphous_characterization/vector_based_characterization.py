"""
3 Vector Analysis (3VA) Extended
================================

3VA is a method for characterizing amorphous materials by analyzing the angles between three vectors in
a diffraction pattern. It is particularly useful for identifying the symmetry of a material but can also be
useful for identifying "fingerprints" of different crystal structures when you have overlapping structures.

The main idea is that even with many overlapping structures, the angles between the vectors will remain
constant.


Why 3 and not 2 vectors?
------------------------
Let's break this questions down. Let's say you have 2 structures, one with 6-fold symmetry and one
with 4-fold symmetry. And both diffraction patterns are overlapping. There are a total of 10 vectors with
the number of combinations of 2 vectors being. nC2 = n!/(n-2)!/2! = n(n-1)/2 = 10(9)/2 = 45 combinations.

Most of these combinations are useless as they are between the 4-fold and 6-fold vectors. But in general the
high symmetry features will win out in the end and you will get a nice visual of the symmetries in the
dataset.
"""

from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pyxem as pxm
from scipy.ndimage import gaussian_filter


all_angles = []
for i in range(100):
    six_fold_angles = (
        (np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 7)[:-1]
    ) % (2 * np.pi)
    four_fold_angles = (
        (np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 5)[:-1]
    ) % (2 * np.pi)

    new_angles = np.hstack((six_fold_angles, four_fold_angles))
    all_2_angles = [
        (
            np.abs(c[1] - c[0])
            if np.abs(c[1] - c[0]) < np.pi
            else (np.pi * 2) - np.abs(c[1] - c[0])
        )
        for c in combinations(new_angles, 2)
    ]
    all_angles = np.hstack((all_angles, all_2_angles))

plt.hist(all_angles, bins=90, range=(0, np.pi))

# You might say that looks pretty good (and it does) but the reality is that only about 5% of diffraction patterns
# are "High Symmetry" and give you the information you want about some disordered structure.
# So what we are more than likely looking at is something like:

all_angles = []
for i in range(100):
    four_fold_angles = (
        (np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 5)[:-1]
    ) % (2 * np.pi)
    two_fold = [
        ((np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 3)[:-1])
        % (2 * np.pi)
        for i in range(19)
    ]

    new_angles = np.hstack((four_fold_angles, *two_fold))
    all_2_angles = [
        (
            np.abs(c[1] - c[0])
            if np.abs(c[1] - c[0]) < np.pi
            else (np.pi * 2) - np.abs(c[1] - c[0])
        )
        for c in combinations(new_angles, 2)
    ]
    all_angles = np.hstack((all_angles, all_2_angles))

plt.hist(all_angles, bins=90, range=(0, np.pi))

# %%
# Now we can barely see the 4-fold symmetry in the histogram, and we have many other angles that are more
# than likely useless... And this is for a "perfect dataset" which will almost never happen in real life.
# So let's look at using 3 vectors.


def get_reduced_angle(vector1, vector2):
    if np.abs((vector2 - vector1)) < np.pi:
        return np.abs(vector2 - vector1)
    else:
        return (np.pi * 2) - np.abs(vector2 - vector1)


all_angles = [[0, 0]]
for i in range(100):
    four_fold_angles = (
        (np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 5)[:-1]
    ) % (2 * np.pi)
    two_fold = [
        ((np.random.random() * np.pi * 2) + np.linspace(0, np.pi * 2, 3)[:-1])
        % (2 * np.pi)
        for i in range(19)
    ]

    new_angles = np.hstack((four_fold_angles, *two_fold))
    all_3_angles = np.array(
        [
            [get_reduced_angle(c[0], c[1]), get_reduced_angle(c[1], c[2])]
            for c in combinations(new_angles, 3)
        ]
    )
    all_angles = np.vstack((all_angles, all_3_angles))

plt.hist2d(all_angles[:, 0], all_angles[:, 1], bins=180)

# or just getting the diagonal of the matrix

bins, x, y = np.histogram2d(all_angles[:, 0], all_angles[:, 1], bins=180)
plt.bar(np.linspace(0, np.pi, 180), [bins[i, i] for i in range(180)], width=np.pi / 180)

# %%
# That's better!  This 2D histogram representation is nice because it is a nice "fingerprint" of all
# the structures in the diffraction pattern. Let's try it for something like an Al nano crystal.

from pyxem.data.simulated_fe import fe_fcc_phase
from orix.quaternion import Rotation
from diffsims.generators.simulation_generator import SimulationGenerator

random_rotations = Rotation.random(5000)

gen = SimulationGenerator()

simulations = gen.calculate_diffraction2d(
    fe_fcc_phase(),
    rotation=random_rotations,
    max_excitation_error=0.05,
    with_direct_beam=False,
)
all_angles = [[0, 0]]

for i in range(1000):
    overlap_vect = np.vstack(
        [simulations.coordinates[i * 5 + j].data[:, :2] for j in range(5)]
    )
    polar = np.arctan2(overlap_vect[:, 0], overlap_vect[:, 1])
    all_3_angles = np.array(
        [
            [get_reduced_angle(c[0], c[1]), get_reduced_angle(c[1], c[2])]
            for c in combinations(polar, 3)
        ]
    )
    all_angles = np.vstack((all_angles, all_3_angles))

plt.figure()
plt.hist2d(
    all_angles[:, 0],
    all_angles[:, 1],
    bins=175,
    range=((0, np.pi * (175 / 180)), (0, np.pi * (175 / 180))),
)

# %%
# This is maybe a little more interesting, we can see a strong 4 Fold symmetry in the histogram.
# But it's maybe not as useful as we'd like.  We can also make a 4D Fingerprint for the dataset by
# including information about the magnitude of the vectors.

all_angles = [[0, 0]]
all_mags = [[0, 0]]

for i in range(100):
    overlap_vect = np.vstack(
        [simulations.coordinates[i * 5 + j].data[:, :2] for j in range(5)]
    )
    polar = np.arctan2(overlap_vect[:, 0], overlap_vect[:, 1])
    mag = np.linalg.norm(overlap_vect[:, :1], axis=1)
    all_3_angles = np.array(
        [
            [get_reduced_angle(c[0], c[1]), get_reduced_angle(c[1], c[2])]
            for c in combinations(polar, 3)
        ]
    )
    mag_difference = np.array(
        [[np.abs(c[0] - c[1]), np.abs(c[1] - c[2])] for c in combinations(mag, 3)]
    )
    all_angles = np.vstack((all_angles, all_3_angles))
    all_mags = np.vstack((all_mags, mag_difference))

all_3_vectors = np.hstack((all_mags, all_angles))
arr, axes = np.histogramdd(all_3_vectors, bins=(40, 40, 180, 180))

hs.signals.Signal2D(arr).isig[:175, :175].plot(
    vmin=4
)  # get rid of 2 fold to help with scaling

# %%
# Trying with some Data:
# ----------------------
# I'm not sure exactly how useful that 4D "fingerprint" is but it __does__ seem to be an interesting data
# representation. Let's take a PdNiP glass and then try to get it's 4D fingerprint.
# Let's start by defining a new function:


def get_angles_and_mag_difference(vectors):
    """Get the angles for each 3 vectors in the dataset as well as the difference in angles.

    This will return a list of vectors at each probe position with dimensions [angle 1, angle 2, diff v1-v2, diff v2-3]
    """
    three_angle_vectors = []
    for combo in combinations(vectors, 3):
        angle_1 = get_reduced_angle(combo[0][1], combo[1][1])
        angle_2 = get_reduced_angle(combo[1][1], combo[2][1])
        angle_3 = get_reduced_angle(combo[2][1], combo[0][1])

        mag_1 = np.abs(combo[0][0] - combo[1][0])
        mag_2 = np.abs(combo[1][0] - combo[2][0])
        mag_3 = np.abs(combo[2][0] - combo[0][0])

        min_angle_indexes = np.argsort([angle_1, angle_2, angle_3])
        angles = np.array([angle_1, angle_2, angle_3])[min_angle_indexes[:2]]
        mags = np.array([mag_1, mag_2, mag_3])[min_angle_indexes[:2]]
        three_angle_vectors.append(np.hstack((mags, angles)))
    three_angle_vectors = np.array(three_angle_vectors)
    if len(three_angle_vectors) == 0:
        three_angle_vectors = np.empty((0, 4))
    return np.array(three_angle_vectors)


s = pxm.data.pdnip_glass(allow_download=True)
s.axes_manager.signal_axes[0].offset = -23.7
s.axes_manager.signal_axes[1].offset = -19.3

s.filter(gaussian_filter, sigma=(1, 1, 0, 0), inplace=True)  # only in real space
s.template_match_disk(disk_r=5, subtract_min=False, inplace=True)

vectors = s.get_diffraction_vectors(threshold_abs=0.5, min_distance=3)

pol = vectors.to_polar()
angs = pol.map(get_angles_and_mag_difference, inplace=False)

arr, axes = np.histogramdd(
    angs.flatten_diffraction_vectors().data[:, 2:],
    bins=(10, 10, 45, 45),
    range=((0, 10), (0, 10), (0, np.pi), (0, np.pi)),
)

histogram_finger_print = hs.signals.Signal2D(arr)
histogram_finger_print.axes_manager.navigation_axes.set(
    scale=1, name=("difference V1<->V2", "difference V2<->V3")
)
histogram_finger_print.axes_manager.signal_axes.set(
    scale=(np.pi / 45), units="rad", name=("$\Delta \phi_1$", "$\Delta \phi_2$")
)


hs.plot.plot_images(
    [
        histogram_finger_print.inav[0, 0],
        histogram_finger_print.inav[2, 2],
        histogram_finger_print.inav[5, 5],
    ]
)


# %%
# Final Thoughts
# --------------
# Please let me (cfrancis@directelectron.com) know if you have any questions or comments about this.
# This is a new method and I would love to hear your feedback on it. The major trick to this is that
# it's necessary to have lots of data (the more, the better) to really start to identify real structures
# in the data and separate them from the random overlaps.
#
