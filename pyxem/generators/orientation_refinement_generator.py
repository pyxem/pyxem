# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""
Generating subpixel resolution on diffraction vectors.
"""

import numpy as np

from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.utils.expt_utils import peaks_as_gvectors
from pyxem.utils.subpixel_refinements_utils import _conventional_xc
from pyxem.utils.subpixel_refinements_utils import get_experimental_square
from pyxem.utils.subpixel_refinements_utils import get_simulated_disc

import warnings


class OrientationRefinementGenerator():
    """Generates subpixel refinement of DiffractionVectors.

    Parameters
    ----------
    dp : ElectronDiffraction2D
        The electron diffraction patterns to be refined
    vectors : DiffractionVectors | ndarray
        Vectors (in calibrated units) to the locations of the spots to be
        refined. If given as DiffractionVectors, it must have the same
        navigation shape as the electron diffraction patterns. If an ndarray,
        the same set of vectors is mapped over all electron diffraction
        patterns.

    References
    ----------
    [1] Pekin et al. Ultramicroscopy 176 (2017) 170-176

    """

    def __init__(self, dp, vectors):
        self.dp = dp
        self.vectors_init = vectors
        self.last_method = None
        sig_ax = dp.axes_manager.signal_axes
        self.calibration = [sig_ax[0].scale, sig_ax[1].scale]
        self.center = [sig_ax[0].size / 2, sig_ax[1].size / 2]

     def refine(self, img, result, projector=None, verbose=True, method="least-squares", fit_tol=0.1,
               vary_center=True, vary_scale=True, vary_alphabeta=True, vary_gamma=True, **kwargs):
        """
        Refine the orientations of all solutions in results agains the given image

        img: ndarray
            Image array
        result: IndexingResult object
            Specifications of the solution to be refined
        projector: Projector object, optional
            This keyword should be specified if projector is not already an attribute on Indexer,
            or if a different one should be used
        method: str, optional
            Minimization method to use, should be one of 'nelder', 'powell', 'cobyla', 'least-squares'
        fit_tol: float
            Tolerance for termination. For detailed control, use solver-specific options.
        """
        if not projector:
            projector = self.projector

        f_kws = kwargs.get("kws", None)
        
        def objfunc(params, img):
            cx = params["center_x"].value
            cy = params["center_y"].value
            al = params["alpha"].value
            be = params["beta"].value
            ga = params["gamma"].value
            sc = params["scale"].value
            
            proj = projector.get_projection(al, be, ga)
            pks = proj[:,3:6]
            score = get_score_shape(img, pks, sc, cx, cy)

            return 1e3/(1+score)

        params = lmfit.Parameters()
        params.add("center_x", value=result.center_x, vary=vary_center, min=result.center_x - 2.0, max=result.center_x + 2.0)
        params.add("center_y", value=result.center_y, vary=vary_center, min=result.center_y - 2.0, max=result.center_y + 2.0)
        params.add("alpha", value=result.alpha, vary=vary_alphabeta)
        params.add("beta",  value=result.beta,  vary=vary_alphabeta)
        params.add("gamma", value=result.gamma, vary=vary_gamma)
        params.add("scale", value=result.scale, vary=vary_scale, min=result.scale*0.8, max=result.scale*1.2)
        
        args = img,

        res = lmfit.minimize(objfunc, params, args=args, method=method, tol=fit_tol, kws=f_kws)

        if verbose:
            lmfit.report_fit(res)
                
        p = res.params
        
        alpha, beta, gamma = [round(p[key].value, 4) for key in ("alpha", "beta", "gamma")]
        scale, center_x, center_y = [round(p[key].value, 2) for key in ("scale", "center_x", "center_y")]
        
        proj = projector.get_projection(alpha, beta, gamma)
        pks = proj[:,3:6]
        
        score = round(get_score_shape(img, pks, scale, center_x, center_y), 2)
        
        # print "Score: {} -> {}".format(int(score), int(score))
        
        refined = IndexingResult(score=score,
                                 number=result.number,
                                 alpha=alpha,
                                 beta=beta,
                                 gamma=gamma,
                                 center_x=center_x,
                                 center_y=center_y,
                                 scale=scale,
                                 phase=result.phase)
        
        return refined

    def probability_distribution(self, img, result, projector=None, verbose=True, vary_center=False, vary_scale=True):
        """https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee

        Calculate posterior probability distribution of parameters"""
        import corner
        import emcee

        if not projector:
            projector = self.projector
        
        def objfunc(params, pks, img):
            cx = params["center_x"].value
            cy = params["center_y"].value
            al = params["alpha"].value
            be = params["beta"].value
            ga = params["gamma"].value
            sc = params["scale"].value
            
            proj = projector.get_projection(al, be, ga)
            pks = proj[:,3:6]
            score = get_score_shape(img, pks, sc, cx, cy)
            
            resid = 1e3/(1+score)
            
            # Log-likelihood probability for the sampling. 
            # Estimate size of the uncertainties on the data
            s = params['f']
            resid *= 1 / s
            resid *= resid
            resid += np.log(2 * np.pi * s**2)
            return -0.5 * np.sum(resid)

        params = lmfit.Parameters()
        params.add("center_x", value=result.center_x, vary=vary_center, min=result.center_x - 2.0, max=result.center_x + 2.0)
        params.add("center_y", value=result.center_y, vary=vary_center, min=result.center_y - 2.0, max=result.center_y + 2.0)
        params.add("alpha", value=result.alpha + 0.01, vary=True, min=result.alpha - 0.1, max=result.alpha + 0.1)
        params.add("beta",  value=result.beta + 0.01,  vary=True, min=result.beta - 0.1, max=result.beta + 0.1)
        params.add("gamma", value=result.gamma + 0.01, vary=True, min=result.gamma - 0.1, max=result.gamma + 0.1)
        params.add("scale", value=result.scale, vary=vary_scale, min=result.scale*0.8, max=result.scale*1.2)
        
        # Noise parameter
        params.add('f', value=1, min=0.001, max=2)
        
        pks_current = projector.get_projection(result.alpha, result.beta, result.gamma)[:,3:5]
        
        args = pks_current, img
        
        mini = lmfit.Minimizer(objfunc, params, fcn_args=args)
        res = mini.emcee(params=params)

        if verbose:
            print("\nMedian of posterior probability distribution")
            print("--------------------------------------------")
            lmfit.report_fit(res)

        # find the maximum likelihood solution
        highest_prob = np.argmax(res.lnprob)
        hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
        mle_soln = res.chain[hp_loc]
        
        if verbose:
            for i, par in enumerate(res.var_names):
                params[par].value = mle_soln[i]
            print("\nMaximum likelihood Estimation")
            print("-----------------------------")
            print(params)
        
        corner.corner(res.flatchain, labels=res.var_names, truths=[res.params[par].value for par in res.params if res.params[par].vary])
        plt.show()
