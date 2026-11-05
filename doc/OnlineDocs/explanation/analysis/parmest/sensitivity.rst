.. _parmest_sensitivity_section:

Global Sensitivity Analysis
===========================

Global sensitivity analysis (GSA) determines how variation in the model outputs is
attributed to the model parameters, :math:`\boldsymbol{\theta}`, when the parameters are
varied over a bounded range. Unlike the local sensitivities used for the covariance matrix
(see :ref:`parmest_covariance_section` Section), GSA varies all parameters at once over the
whole range. In parmest, GSA can be performed using partial rank correlation coefficients
(PRCC), the Morris method, or Sobol' indices. These methods are compared in [Gas26]_.

In parmest, GSA is performed on the labeled model of each experiment, not on the extensive
form model used for parameter estimation. For each sample of :math:`\boldsymbol{\theta}`, the
unknown parameters are fixed at the sampled values, the labeled model is solved, and the
values of the experiment outputs are recorded. The data are not used, so GSA can be
performed before or after parameter estimation. Each method returns a ``pandas.DataFrame``
indexed by experiment number, output name, and parameter name.

Each varied parameter is sampled uniformly between its bounds. By default, the bounds of the
parameter variable in the labeled model are used. Bounds can also be supplied with the
``theta_bounds`` argument, e.g., to vary each parameter by :math:`\pm 25\%` of its nominal
value. A subset of the parameters can be varied with the ``theta_names`` argument; the other
parameters stay at their values in the labeled model. Solves that do not terminate optimally
are left out of the analysis and a warning is logged.

By default (``warmstart="none"``), every solve starts from the variable values of the labeled
model, so each sample is solved independently and the results do not depend on the sample
order. With ``warmstart="previous"``, each solve starts from the last optimal solution, which
can be faster. However, if the model has more than one solution for the same parameter values
(e.g., a bistable steady state), the solution found can then depend on the previous sample.
For such models, initialize the labeled model at the state of interest and keep the default.

.. list-table:: Global sensitivity analysis methods in parmest
   :header-rows: 1
   :widths: 20 20 30 30

   * - Method
     - Type
     - Model solves per experiment
     - Results
   * - PRCC
     - Ranking
     - :math:`N`
     - Strength and direction of monotonic effects
   * - Morris
     - Screening
     - :math:`r (k + 1)`
     - Mean and spread of the elementary effects
   * - Sobol'
     - Variance-based
     - :math:`N (k + 2)`, or :math:`N (2k + 2)` with second order indices
     - Fraction of the output variance due to each parameter and their interactions

Here :math:`k` is the number of varied parameters, :math:`N` is the number of samples, and
:math:`r` is the number of Morris trajectories. Because Sobol' indices are the most
expensive, a common workflow is to screen the parameters with the Morris method first and
then compute the Sobol' indices of the important parameters [Gas26]_.

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0],
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create the Experiment class
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

Partial Rank Correlation Coefficient
------------------------------------

PRCC is a ranking method that measures the strength and direction of a monotonic
relationship between each parameter and each output [MHR+08]_. The parameter samples and
outputs are rank transformed. The PRCC between parameter :math:`\theta_j` and output
:math:`y` is the correlation between the residuals of the linear regressions of
:math:`\theta_j` and :math:`y` on the other parameters:

.. math::
   \rho_{\theta_j y} = \frac{\text{Cov}\left(\theta_j - \hat{\theta}_j, y - \hat{y}\right)}
    {\sqrt{\text{Var}\left(\theta_j - \hat{\theta}_j\right) \text{Var}\left(y - \hat{y}\right)}}

A PRCC near :math:`\pm 1` indicates a strong effect and a PRCC near 0 indicates little effect.
The p-value of the test :math:`\rho = 0` is also returned. PRCC assumes that each output
changes monotonically with each parameter, and that the sampled parameters are independent.
By default, the parameters are sampled with a Latin hypercube design optimized to reduce the
correlation between parameters (``lhs_optimization="random-cd"``), and a warning is logged if
two sampled parameters have a rank correlation above 0.1.

Outputs that differ by less than the solver tolerance can be rounded before ranking with
``output_decimals``, so they are ties rather than arbitrary ranks. ``partial=False`` gives the
Spearman rank correlation coefficient instead, which does not remove the effect of the other
parameters and is therefore smaller in magnitude when several parameters affect the output.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> theta_bounds = {"asymptote": (10.0, 30.0), "rate_constant": (0.2, 1.0)}
    >>> prcc = pest.sensitivity_prcc(n_samples=20, theta_bounds=theta_bounds, seed=0)

Morris Method
-------------

The Morris method is a screening method that identifies the parameters with little effect on
the outputs [Morris91]_. Each parameter is scaled to :math:`[0, 1]` and divided into :math:`p`
levels. Each of the :math:`r` trajectories starts at a random point on this grid and changes one
parameter at a time by :math:`\Delta = p / (2 (p - 1))`. The elementary effect of parameter
:math:`\theta_j` is

.. math::
   EE_j = \frac{y(\theta_1, \ldots, \theta_j + \Delta, \ldots, \theta_k) - y(\theta_1, \ldots,
    \theta_k)}{\Delta}

The mean of the elementary effects, :math:`\mu_j`, the mean of their absolute values,
:math:`\mu^*_j`, and their standard deviation, :math:`\sigma_j`, are returned. A small
:math:`\mu^*_j` means the parameter has little effect, and a large :math:`\sigma_j` means its
effect is nonlinear or depends on the other parameters. A parameter is marked as important for
an output if :math:`\mu^*_j` is at least ``screening_threshold`` (default 0.5) times the largest
:math:`\mu^*` for that output.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> morris = pest.sensitivity_morris(
    ...     n_trajectories=5, theta_bounds=theta_bounds, seed=0
    ... )
    >>> important = list(morris[morris["important"]].index.unique("parameter"))

Sobol' Indices
--------------

Sobol' indices decompose the variance of an output into contributions from each parameter
and their interactions [SAA+10]_. The first order index is the fraction of the output variance
due to parameter :math:`\theta_j` alone, the total order index also includes the interactions
of :math:`\theta_j` with the other parameters, and the second order index is the fraction due to
the interaction of :math:`\theta_j` and :math:`\theta_l`:

.. math::
   S_{1,j} = \frac{\text{Var}_{\theta_j}\left(\text{E}_{\boldsymbol{\theta}_{\sim j}}
    (y \mid \theta_j)\right)}{\text{Var}(y)}, \quad
   S_{T,j} = \frac{\text{E}_{\boldsymbol{\theta}_{\sim j}}\left(\text{Var}_{\theta_j}
    (y \mid \boldsymbol{\theta}_{\sim j})\right)}{\text{Var}(y)}, \quad
   S_{2,jl} = \frac{\text{Var}_{\theta_j \theta_l}\left(\text{E}_{\boldsymbol{\theta}_{\sim jl}}
    (y \mid \theta_j, \theta_l)\right)}{\text{Var}(y)} - S_{1,j} - S_{1,l}

where :math:`\boldsymbol{\theta}_{\sim j}` are all parameters except :math:`\theta_j`. The
indices are estimated with the Saltelli sample design built from a scrambled Sobol' sequence.
A power of 2 for ``n_samples`` gives the most even coverage. Below, the Sobol' indices are
computed for the parameters kept by the Morris screening above.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> sobol = pest.sensitivity_sobol(
    ...     n_samples=8, theta_names=important, theta_bounds=theta_bounds, seed=0
    ... )

Second order indices are returned in a second ``DataFrame`` when ``calc_second_order=True``.

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> sobol, sobol_second_order = pest.sensitivity_sobol(
    ...     n_samples=8, calc_second_order=True, theta_bounds=theta_bounds, seed=0
    ... )
