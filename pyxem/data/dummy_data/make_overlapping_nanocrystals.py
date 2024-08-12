from diffsims.generators.rotation_list_generators import get_beam_directions_grid
from scipy.spatial import ConvexHull
from matplotlib.collections import PolyCollection
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.quaternion import Rotation

import matplotlib.pyplot as plt
import numpy as np
from random import random
from math import cos, sin, pi, radians
from skimage.segmentation import watershed
from copy import deepcopy
from skimage.draw import disk


class CrystalSTEMSimulation:
    """
    A class for creating a 4-D STEM dataset with overlapping nanocrystals.

    Parameters
    ----------
    phase : orix.crystal_map.Phase
        The phase to simulate the diffraction data for.
    num_crystals : int, optional
        The number of nanocrystals to simulate.
    k_range : float, optional
        The range of reciprocal space to simulate.
    crystal_size : int, optional
        The size of the nanocrystals in pixels.
    real_space_pixels : int, optional
        The number of pixels in the real space image to simulate.
    recip_space_pixels : int, optional
        The number of pixels in the reciprocal space image to simulate.
    generator : SimulationGenerator, optional
        The SimulationGenerator to use for calculating the diffraction data.
    max_excitation_error : float, optional
        The maximum excitation error to use for the diffraction data.
    rotations : Rotation, optional
        The rotations to use for the nanocrystals.
    accelerating_voltage : float, optional
        The accelerating voltage to use for the diffraction data.


    Notes
    -----
    This function is not optimized for speed and is intended for generating
    small datasets for testing purposes and for comparing the performance of
    different algorithms.
    """

    def __init__(
        self,
        phase,
        num_crystals=10,
        k_range=1,
        crystal_size=10,
        real_space_pixels=128,
        recip_space_pixels=64,
        generator=None,
        max_excitation_error=0.015,
        rotations=None,
        accelerating_voltage=200,
    ):
        # Set all the parameters
        if generator is None:
            self.generator = SimulationGenerator(
                accelerating_voltage=accelerating_voltage, minimum_intensity=1e-10
            )
        else:
            self.generator = generator
        self.num_crystals = num_crystals
        self.k_range = k_range
        self.crystal_size = crystal_size
        self.real_space_pixels = real_space_pixels
        self.recip_space_pixels = recip_space_pixels
        self.phase = phase

        # Set up the Simulations for calculating the Diffraction Data
        if rotations is None:
            beam_dir = get_beam_directions_grid("cubic", resolution=1)
            r = Rotation.from_euler(beam_dir, degrees=True, direction="crystal2lab")
        else:
            r = rotations

        self.simulation = self.generator.calculate_diffraction2d(
            phase=phase,
            reciprocal_radius=1,
            rotation=r,
            max_excitation_error=max_excitation_error,
        )

        # Randomly set up the center and rotations for the crystals
        self.centers = np.random.randint(
            low=crystal_size + 5,
            high=real_space_pixels - (crystal_size + 5),
            size=(num_crystals, 2),
        )
        self.random_rotations = np.random.randint(0, len(r.data), num_crystals)
        self.random_inplane = np.random.random(num_crystals) * np.pi

        # Get the Nano crystal Vectors

        (
            self.vectors,
            self.coordinates,
            self.real_space_vectors,
            self.intensities,
        ) = self.get_nano_crystal_vectors()

    def get_nano_crystal_vectors(self):
        """
        Get the nano crystal vectors

        """
        real_space_pixels = self.real_space_pixels
        coordinates = []
        real_space_pos = []
        vectors = []
        intens = []
        # For each crystal randomly rotate a
        for rot, center, inplane in zip(
            self.random_rotations, self.centers, self.random_inplane
        ):
            # Getting the simulation
            cor = self.simulation.irot[rot].coordinates.data[:, :2]
            inten = self.simulation.irot[rot].coordinates.intensity

            vector_pos = np.argwhere(
                create_blob(
                    center=center,
                    size=self.crystal_size,
                    real_space_pixels=real_space_pixels,
                )
            )
            real_space_pos.append(vector_pos)

            # Randomly rotating in plane
            corx = cor[:, 0] * np.cos(inplane) - cor[:, 1] * np.sin(inplane)
            cory = cor[:, 0] * np.sin(inplane) + cor[:, 1] * np.cos(inplane)
            cors = np.stack([corx, cory, inten], axis=1)
            coordinates.append(cors)
            intens.append(inten)
            # Create a list of 4-D Vectors describing the extent of the
            v = [list(v) + list(c) for v in vector_pos for c in cors]
            vectors.append(v)
            intens.append(intens)
        # Unpack the vectors for plotting
        vectors = np.array([v for vector in vectors for v in vector])
        return vectors, coordinates, real_space_pos, intens

    def get_coords(self, low=0.0, high=0.9):
        filtered_coords = []
        for co in self.coordinates:
            norms = np.linalg.norm(co[:, :2], axis=1)
            within_range = (norms > low) * (norms < high)
            filtered_coords.append(co[within_range])
        return filtered_coords

    def make_4d_stem(
        self,
        num_electrons=1,
        electron_gain=1,
        noise_level=0.1,
        radius=5,
    ):
        """
        Make a 4-D Dataset

        Parameters
        ----------
        num_electrons : int, optional
            The number of electrons per pixel on average
        electron_gain : float, optional
            The gain of the detector. Each electron will be multiplied by this value
        noise_level : float, optional
            The level of noise to add to the dataset. Noise will fall between 0 and this value
        radius : int, optional
            The radius of the disks to add to the dataset

        Returns
        -------
        np.ndarray
            A 4-D array representing the 4-D STEM dataset
        """
        real_space_pixels = self.real_space_pixels
        recip_space_pixels = self.recip_space_pixels
        k_range = self.k_range

        vectors_by_index = convert_to_by_index(
            self.vectors, real_space=self.real_space_pixels
        )

        arr = np.zeros(
            (
                real_space_pixels,
                real_space_pixels,
                recip_space_pixels,
                recip_space_pixels,
            )
        )

        scale = (recip_space_pixels / 2) / k_range
        center = recip_space_pixels / 2
        for i in np.ndindex((real_space_pixels, real_space_pixels)):
            vector = vectors_by_index[i]
            im = np.zeros((self.recip_space_pixels, self.recip_space_pixels))
            for v in vector:
                im = add_disk(
                    im, center=v[:2] * scale + center, radius=radius, intensity=v[2]
                )
            arr[i] = im

        arr = arr * num_electrons
        arr = np.random.poisson(arr) * electron_gain

        noise = np.random.random(arr.shape) * noise_level
        arr = arr + noise
        return arr

    def plot_real_space(
        self,
        ax=None,
        remove_below_n=None,
        remove_non_symmetric=False,
        **kwargs,
    ):
        """
        Plot the real space image of the nanocrystals. This is a helpful ground truth


        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes to plot the image on
        remove_below_n: int
            Remove nanocrystals with less than n k-vectors. This is helpful for removing nanocrystals
            that don't have many characteristic k-vectors
        remove_non_symmetric: bool
            Remove nanocrystals that have non-symmetric k-vectors. This is helpful for removing nanocrystals
            that don't have symmetric diffraction
        kwargs:
            Additional keyword arguments to pass to the `matplotlib.collections.PolyCollection`
            class.

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        colors = [
            "blue",
            "green",
            "red",
            "yellow",
            "purple",
            "orange",
            "violet",
            "indigo",
            "black",
            "grey",
        ]
        verts = []
        if remove_below_n is not None:
            vectors = [
                v
                for v, c in zip(self.real_space_vectors, self.coordinates)
                if len(c) > remove_below_n
            ]
        else:
            vectors = self.real_space_vectors

        if remove_non_symmetric:
            new_vectors = []
            for c, v in zip(self.coordinates, self.real_space_vectors):
                un, counts = np.unique(
                    np.round(np.linalg.norm(c[:, :2], axis=1), 1), return_counts=True
                )
                if np.any(counts > 2):
                    new_vectors.append(v)
            vectors = new_vectors

        for v in vectors:
            hull = ConvexHull(v)
            vert = hull.points[hull.vertices]
            verts.append(vert)
        p = PolyCollection(verts, color=colors, **kwargs)
        ax.add_collection(p)
        ax.set_xlim(0, self.real_space_pixels)
        ax.set_ylim(0, self.real_space_pixels)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_example_dp(
        self,
        rotation_ind=0,
        num_electrons=1,
        electron_gain=1,
        noise_level=0.1,
        pixels=64,
        reciprocal_radius=1,
        ax=None,
        threshold=0.8,
        disk_r=5,
        **kwargs,
    ):
        """
        Plot an example diffraction pattern

        Parameters
        ----------
        rotation_ind:
            The index of the rotation to plot
        num_electrons:
            The number of electrons per pixel on average
        electron_gain:
            The gain of the detector. Each electron will be multiplied by this value
        noise_level:
            The level of noise to add to the dataset. Noise will fall between 0 and this value
        pixels:
            The number of pixels in the diffraction pattern
        reciprocal_radius:
            The reciprocal radius to plot
        ax: matplotlib.axes.Axes, optional
            The axes to plot the image on
        threshold:
            The threshold to use for the image
        disk_r: int
            The radius of the disks to add to the dataset
        kwargs:
            Additional keyword arguments to pass to the `matplotlib.pyplot.imshow`
            function.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        sim = self.simulation.irot[rotation_ind]
        img = np.zeros((pixels, pixels))
        coors = sim.coordinates.data[:, :2]
        intens = sim.coordinates.intensity
        scale = (pixels / 2) / reciprocal_radius
        center = pixels / 2

        for c, inten in zip(coors, intens):
            img = add_disk(
                img, center=c * scale + center, radius=disk_r, intensity=inten
            )
        img = img * num_electrons

        img = np.random.poisson(img) * electron_gain
        img = img + np.random.random(img.shape) * noise_level
        if threshold is not None:
            kwargs["vmax"] = np.max(img) * threshold
        ax.imshow(
            img,
            extent=(
                -reciprocal_radius,
                reciprocal_radius,
                -reciprocal_radius,
                reciprocal_radius,
            ),
            **kwargs,
        )
        return ax


def create_blob(center, size, real_space_pixels):
    """
    A utility function to create a blob in the real space image

    Parameters
    ----------
    center:
        The center of the blob
    size: int
        The "radius" of the blob in pixels
    real_space_pixels:
        The number of pixels in the real space image to account for
    """
    N = 4
    amps = [random() * (1 / (2 * N)) for _ in range(N)]
    phases = [random() * 2 * pi for _ in range(N)]

    points = np.empty((360, 2))
    img = np.zeros((real_space_pixels, real_space_pixels))
    for deg in range(360):
        alpha = radians(deg)
        radius = 1 + sum([amps[i] * cos((i + 1) * alpha + phases[i]) for i in range(N)])
        points[deg, 0] = (cos(alpha) * radius * size) + center[0]
        points[deg, 1] = (sin(alpha) * radius * size) + center[1]
    points = np.array(
        np.round(
            points,
        ),
        dtype=int,
    )
    img[points[:, 0], points[:, 1]] = 1
    img = watershed(img) == 2
    return img


def convert_to_by_index(peaks, real_space):
    """
    Convert  a vector of vectors to a indexable array of vectors in real space for
    creating a 4-D STEM dataset.

    Parameters
    ----------
    peaks: np.ndarray
        The array of vectors
    real_space:
        The number of pixels in the real space image to account for
    """
    new_peaks = deepcopy(peaks)
    ind = np.lexsort((new_peaks[:, 1], new_peaks[:, 0]))
    sorted_peaks = new_peaks[ind]
    by_ind_peaks = np.empty((real_space, real_space), dtype=object)
    low_x_ind = np.searchsorted(sorted_peaks[:, 0], range(0, real_space), side="left")
    high_x_ind = np.searchsorted(
        sorted_peaks[:, 0], range(1, real_space + 1), side="left"
    )
    for i, (lo_x, hi_x) in enumerate(zip(low_x_ind, high_x_ind)):
        x_inds = sorted_peaks[lo_x:hi_x]
        low_y_ind = np.searchsorted(x_inds[:, 1], range(0, real_space), side="left")
        high_y_ind = np.searchsorted(
            x_inds[:, 1], range(1, real_space + 1), side="left"
        )
        for j, (lo_y, hi_y) in enumerate(zip(low_y_ind, high_y_ind)):
            by_ind_peaks[i, j] = x_inds[lo_y:hi_y, 2:]
    return by_ind_peaks


def add_disk(image, center, radius, intensity):
    """
    Add a disk to an image

    Parameters
    ----------
    image: np.ndarray
        The image to add the disk to
    center:
        The center of the disk
    radius:
        The radius of the disk
    intensity:
        The intensity of the disk

    """
    disk_image = np.zeros_like(image)
    rr, cc = disk(center=center, radius=radius, shape=image.shape)
    disk_image[rr, cc] = intensity  # expected 1 electron per pixel
    image = disk_image + image
    return image
