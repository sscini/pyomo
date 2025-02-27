#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import subprocess
import sys

import pyomo.common.unittest as unittest
import pyomo.environ as pe

from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.base import TerminationCondition


opt = Highs()
if not opt.available():
    raise unittest.SkipTest


class TestBugs(unittest.TestCase):
    def test_mutable_params_with_remove_cons(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()

        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)

        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + m.p1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + m.p2)

        m.p1.value = 1
        m.p2.value = 1

        opt = Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)

        del m.c1
        m.p2.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -8)

    def test_mutable_params_with_remove_vars(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)

        m.y.setlb(m.p1)
        m.y.setub(m.p2)

        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)

        m.p1.value = -10
        m.p2.value = 10

        opt = Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, 1)

        del m.c1
        del m.c2
        m.p1.value = -9
        m.p2.value = 9
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -9)

    def test_fix_and_unfix(self):
        # Tests issue https://github.com/Pyomo/pyomo/issues/3127

        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.Binary)
        m.y = pe.Var(domain=pe.Binary)
        m.fx = pe.Var(domain=pe.NonNegativeReals)
        m.fy = pe.Var(domain=pe.NonNegativeReals)
        m.c1 = pe.Constraint(expr=m.fx <= m.x)
        m.c2 = pe.Constraint(expr=m.fy <= m.y)
        m.c3 = pe.Constraint(expr=m.x + m.y <= 1)

        m.obj = pe.Objective(expr=m.fx * 0.5 + m.fy * 0.4, sense=pe.maximize)

        opt = Highs()

        # solution 1 has m.x == 1 and m.y == 0
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.best_feasible_objective, 0.5, places=5)

        # solution 2 has m.x == 0 and m.y == 1
        m.y.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 0, places=5)
        self.assertAlmostEqual(m.fy.value, 1, places=5)
        self.assertAlmostEqual(r.best_feasible_objective, 0.4, places=5)

        # solution 3 should be equal solution 1
        m.y.unfix()
        m.x.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.best_feasible_objective, 0.5, places=5)

    def test_capture_highs_output(self):
        # tests issue #3003
        #
        # Note that the "Running HiGHS" message is only emitted the
        # first time that a model is instantiated.  We need to test this
        # in a subprocess to trigger that output.
        model = [
            'import pyomo.environ as pe',
            'm = pe.ConcreteModel()',
            'm.x = pe.Var(domain=pe.NonNegativeReals)',
            'm.y = pe.Var(domain=pe.NonNegativeReals)',
            'm.obj = pe.Objective(expr=m.x + m.y, sense=pe.maximize)',
            'm.c1 = pe.Constraint(expr=m.x <= 10)',
            'm.c2 = pe.Constraint(expr=m.y <= 5)',
            'from pyomo.contrib.appsi.solvers.highs import Highs',
            'result = Highs().solve(m)',
            'print(m.x.value, m.y.value)',
        ]

        with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
            subprocess.run([sys.executable, '-c', ';'.join(model)])
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(OUT.getvalue(), "10.0 5.0\n")

        model[-2:-1] = [
            'opt = Highs()',
            'opt.config.stream_solver = True',
            'result = opt.solve(m)',
        ]
        with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
            subprocess.run([sys.executable, '-c', ';'.join(model)])
        self.assertEqual(LOG.getvalue(), "")
        # This is emitted by the model set-up
        self.assertIn("Running HiGHS", OUT.getvalue())
        # This is emitted by the solve()
        self.assertIn("HiGHS run time", OUT.getvalue())
        ref = "10.0 5.0\n"
        self.assertEqual(ref, OUT.getvalue()[-len(ref) :])

    def test_warm_start(self):
        m = pe.ConcreteModel()

        # decision variables
        m.x1 = pe.Var(domain=pe.Integers, name="x1", bounds=(0, 10))
        m.x2 = pe.Var(domain=pe.Reals, name="x2", bounds=(0, 10))
        m.x3 = pe.Var(domain=pe.Binary, name="x3")

        # objective function
        m.OBJ = pe.Objective(expr=(3 * m.x1 + 2 * m.x2 + 4 * m.x3), sense=pe.maximize)

        # constraints
        m.C1 = pe.Constraint(expr=m.x1 + m.x2 <= 9)
        m.C2 = pe.Constraint(expr=3 * m.x1 + m.x2 <= 18)
        m.C3 = pe.Constraint(expr=m.x1 <= 7)
        m.C4 = pe.Constraint(expr=m.x2 <= 6)

        # MIP start
        m.x1 = 4
        m.x2 = 4.5
        m.x3 = True

        # solving process
        with Highs() as opt, capture_output() as output:
            opt.solve(m, tee=True, warmstart=True)
        log = output.getvalue()
        self.assertIn("MIP start solution is feasible, objective value is 25", log)
