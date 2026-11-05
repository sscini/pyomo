# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import pandas as pd
from os.path import join, abspath, dirname
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
    ReactorDesignExperiment,
)


def main():

    # Read in data
    file_dirname = dirname(abspath(str(__file__)))
    file_name = abspath(join(file_dirname, "reactor_data.csv"))
    data = pd.read_csv(file_name)

    # Create an experiment list. Two experiments keep the example fast.
    exp_list = [ReactorDesignExperiment(data, i) for i in range(2)]

    pest = parmest.Estimator(exp_list, obj_function="SSE")

    # Vary each rate constant by +/- 25% of its value in the model
    theta_nominal = {"k1": 5.0 / 6.0, "k2": 5.0 / 3.0, "k3": 1.0 / 6000.0}
    theta_bounds = {k: (0.75 * v, 1.25 * v) for k, v in theta_nominal.items()}

    # Partial rank correlation coefficients (ranking, with direction)
    prcc = pest.sensitivity_prcc(n_samples=50, theta_bounds=theta_bounds, seed=0)
    print("PRCC:\n", prcc.to_string())

    # Morris method (screening)
    morris = pest.sensitivity_morris(
        n_trajectories=10, theta_bounds=theta_bounds, seed=0
    )
    print("\nMorris:\n", morris.to_string())

    # Keep the thetas that are important for the outlet concentration of B in
    # any experiment, and compute their Sobol' indices. The other thetas stay
    # at their values in the model.
    morris_cb = morris.xs("cb", level="output")
    important = morris_cb[morris_cb["important"]].index.unique("parameter")
    theta_names = [name for name in theta_nominal if name in important]
    print("\nThetas kept by Morris screening for cb:", theta_names)

    # A larger n_samples reduces the sampling error of the indices
    sobol = pest.sensitivity_sobol(
        n_samples=32, theta_names=theta_names, theta_bounds=theta_bounds, seed=0
    )
    print("\nSobol:\n", sobol.xs("cb", level="output").to_string())

    return prcc, morris, sobol


if __name__ == "__main__":
    main()
