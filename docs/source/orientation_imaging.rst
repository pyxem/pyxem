Orientation Imaging
===================

Orientation imaging can be achieved in a number of ways based on the acquisition
of numerous electron diffraction patterns in the scanning transmission
electron microscope.

Pattern Matching
----------------

Pattern matching, whereby each diffraction pattern in the 4D-S(P)ED dataset is
compared to a library of pre-computed theoretical templates, has become the most
common method of assigning crystal phase and orientation to each recorded pattern.
The implementation described here, and incorporated in pycrystem, follows work by
Rauch \textit{et al.} \cite{Rauch2005} and involves: image pre-processing,
generation of a library of theoretical templates, and determination of the
theoretical template that best fits each pattern.

Diffraction Library Generation
******************************

Generating a library of theoretical templates involves specification of the
crystal phases and orientations to be included, followed by simulation of a
template diffraction pattern corresponding to each phase and orientation. Although
it might be appealing to include all known phases and orientations this will
result in a very slow matching step as well as likely ambiguity between
crystallographically similar structures. In practice, it is preferable to use prior
knowledge to limit the template library to a minimal but sufficient size. The
phases to be included are therefore usually limited to a small number of expected
or likely phases based on previous characterization. The orientations to be
included are most generally selected as a uniform sampling of symmetry inequivalent
orientations corresponding to the crystallography of the given phases. In some
cases, it may be preferable to restrict the included orientations further and pyxem
allows total freedom in this regard. Uniform sampling of orientation space requires
consideration of the topology of orientation space \cite{}. In pyxem, the utility
method, get_equispaced_so3_mesh(), obtains an equispaced grid using even steps in
the Euler angle parameterization.

Simulation of template patterns may, in principle, be achieved using any of the
well-known kinematical or dynamical simulation methods. The complexity justified
depends on the precision and accuracy required as well as the extent to which an
ideal simulation can be properly compared with experimental data. In the pattern
matching framework presented here, a simple kinematical simulation is sufficient
and a library of kinematically simulated templates can be obtained from a list of
phases and orientations using the DiffractionLibraryGenerator() class in pyxem.


Pattern Matching Metric
***********************

Pattern matching between point like geometric templates and experimental spot
diffraction patterns is typically based on a normalized correlation index,
$Q_{i}$, defined as:

\begin{equation} \label{eq:correlation}
Q_{i} = \frac{\sum_{x,y} P(x,y) T_{i}(x,y)}{\sqrt[]{\sum_{x,y}P^{2}(x,y)}\sqrt[]{\sum_{x,y}T_{i}^{2}(x,y)}}
\end{equation}

where $P(x,y)$ is the intensity of the pixel with coordinates (x,y) in the
experimental diffraction pattern and $T_{i}(x,y)$ is the intensity of template i
at (x,y). Efficient evaluation of Equation \ref{eq:correlation} is achieved by
only considering pixels corresponding to non-zero values in the geometric templates.
This metric presents two primary limitations: (1) the range of values is not
normalised between different phases meaning that some phases will typically give
greater scores than others, and (2) the presence of additional diffraction peaks
in the experimental diffraction pattern with respect to the template is not
penalized but the opposite is. Normalisation may be achieved by scaling the
matching results of different phases by a factor. The presence of too many peaks
in the experimental data might in future be penalised by adding an additional
regularisation term.
