#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026 National Technology and Engineering Solutions of
#  Sandia, LLC Under the terms of Contract DE-NA0003525 with National
#  Technology and Engineering Solutions of Sandia, LLC, the U.S. Government
#  retains certain rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import numpy as np, pandas as pd

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

ipopt_available = pyo.SolverFactory("ipopt").available()


class AffineTwoThetaExperiment(Experiment):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta_a = pyo.Var(initialize=1.0, bounds=(0.0, 4.0))
        m.theta_b = pyo.Var(initialize=0.0, bounds=(-3.0, 3.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))
        m.eq = pyo.Constraint(expr=m.y == m.theta_a * m.x + m.theta_b)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            [
                (m.theta_a, pyo.ComponentUID(m.theta_a)),
                (m.theta_b, pyo.ComponentUID(m.theta_b)),
            ]
        )
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class _InitRecordingEstimator(parmest.Estimator):
    """Estimator that records the initial theta values of each profile solve."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_theta_inits = []

    def _Q_opt(self, *args, **kwargs):
        if kwargs.get("fixed_theta_values"):
            self.profile_theta_inits.append(dict(kwargs["theta_vals"]))
        return super()._Q_opt(*args, **kwargs)


def _build_two_theta_estimator(estimator_class=parmest.Estimator):
    # Data lie exactly on y = 2x + 1, so theta_hat = (2, 1) and, with
    # theta_a fixed at a, the profiled optimum is theta_b = 5 - 2a.
    exp_list = [
        AffineTwoThetaExperiment(1.0, 3.0),
        AffineTwoThetaExperiment(2.0, 5.0),
        AffineTwoThetaExperiment(3.0, 7.0),
    ]
    return estimator_class(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestProfileLikelihood(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_contains_baseline_minimum_point(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"]
        self.assertAlmostEqual(res["baseline"]["obj_hat"], prof["obj"].min(), places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_partial_fix_enforced(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"]
        self.assertTrue(np.allclose(prof["theta_value"], prof["theta__theta_a"]))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_other_thetas_unfixed(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"].sort_values("theta_value")
        self.assertGreater(
            prof["theta__theta_b"].max() - prof["theta__theta_b"].min(), 1e-6
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_repeatability_same_seed(self):
        pest = _build_two_theta_estimator()
        res1 = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt", seed=9
        )
        res2 = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt", seed=9
        )
        cols = [
            "theta_value",
            "obj",
            "status",
            "success",
            "theta__theta_a",
            "theta__theta_b",
        ]
        self.assertTrue(res1["profiles"][cols].equals(res2["profiles"][cols]))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_warmstart_neighbor_values_used(self):
        # theta_hat["theta_b"] is deliberately off (0.0 instead of 1.0) so that
        # warm-started initial values are distinguishable from theta_hat.
        theta_hat = {"theta_a": 2.0, "theta_b": 0.0}
        grid = [2.0, 2.1, 2.2]

        pest = _build_two_theta_estimator(_InitRecordingEstimator)
        pest.profile_likelihood(
            "theta_a", grid=grid, theta_hat=theta_hat, obj_hat=0.0, warmstart="neighbor"
        )
        inits = pest.profile_theta_inits
        self.assertEqual([init["theta_a"] for init in inits], grid)
        # First solve starts from theta_hat; each later solve starts from the
        # previous converged point (theta_b = 5 - 2 * theta_a).
        self.assertEqual(inits[0]["theta_b"], 0.0)
        self.assertAlmostEqual(inits[1]["theta_b"], 1.0, places=6)
        self.assertAlmostEqual(inits[2]["theta_b"], 0.8, places=6)

        pest = _build_two_theta_estimator(_InitRecordingEstimator)
        pest.profile_likelihood(
            "theta_a", grid=grid, theta_hat=theta_hat, obj_hat=0.0, warmstart="none"
        )
        self.assertEqual(
            [init["theta_b"] for init in pest.profile_theta_inits], [0.0, 0.0, 0.0]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_failure_recorded_continue(self):
        pest = _build_two_theta_estimator()
        # theta_a = 5.0 is outside the model bounds (0, 4), so that solve
        # fails while the neighboring grid points still succeed.
        with LoggingIntercept(level=logging.WARNING) as LOG:
            res = pest.profile_likelihood(
                "theta_a",
                grid=[1.9, 5.0, 2.1],
                theta_hat={"theta_a": 2.0, "theta_b": 1.0},
                obj_hat=0.0,
            )
        self.assertIn("outside the bounds", LOG.getvalue())

        prof = res["profiles"].set_index("theta_value")
        self.assertEqual(len(prof), 4)
        self.assertIn("exception", str(prof.loc[5.0, "status"]))
        self.assertFalse(bool(prof.loc[5.0, "success"]))
        self.assertTrue(np.isnan(prof.loc[5.0, "obj"]))
        self.assertTrue(prof.drop(index=5.0)["success"].all())

    def test_profile_all_failures_returns_structure(self):
        pest = _build_two_theta_estimator()
        # An unknown solver makes every profile solve raise.
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.0, 2.1],
            theta_hat={"theta_a": 2.0, "theta_b": 0.0},
            obj_hat=1.0,
            solver="not_a_solver",
        )

        prof = res["profiles"]
        self.assertEqual(len(prof), 3)
        self.assertFalse(prof["success"].any())
        self.assertTrue(prof["status"].astype(str).str.contains("exception").all())
        self.assertTrue(prof["obj"].isna().all())

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_user_grid_preserved(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.2, 2.4, 2.0],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
        )

        attempted = res["profiles"]["theta_value"].tolist()
        self.assertEqual(attempted, [1.2, 2.0, 2.4])

    def test_profile_auto_grid_includes_theta_hat(self):
        pest = _build_two_theta_estimator()
        grid = pest._build_profile_grid(
            profiled_theta="theta_a",
            grid=[1.0, 1.5, 2.5],
            n_grid=5,
            theta_hat={"theta_a": 2.0, "theta_b": 0.0},
        )
        self.assertIn(2.0, set(np.round(grid, 12)))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_result_columns_schema(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.0],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
        )
        prof = res["profiles"]
        self.assertTrue(
            set(
                [
                    "profiled_theta",
                    "theta_value",
                    "obj",
                    "delta_obj",
                    "lr_stat",
                    "status",
                    "success",
                    "solve_time",
                    "theta__theta_a",
                    "theta__theta_b",
                ]
            ).issubset(prof.columns)
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_baseline_from_multistart(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            profiled_theta="theta_a",
            n_grid=5,
            use_multistart_for_baseline=True,
            baseline_multistart_kwargs={
                "n_restarts": 3,
                "multistart_sampling_method": "uniform_random",
                "seed": 7,
            },
        )
        self.assertIn("baseline", res)
        self.assertIn("profiles", res)
        self.assertTrue(np.isfinite(res["baseline"]["obj_hat"]))
        self.assertGreaterEqual(res["profiles"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
