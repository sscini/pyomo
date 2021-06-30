#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the canonical expressions
#

import os
from filecmp import cmp
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir

from pyomo.environ import (
    ConcreteModel, Var, Param,  Constraint, Objective,  Block, sin,
    maximize, Binary, Suffix
)

thisdir = this_file_dir()


class Test(unittest.TestCase):

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix+".bar.baseline", prefix+".bar.out"

    def _check_baseline(self, model, **kwds):
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        io_options = {"symbolic_solver_labels": True}
        io_options.update(kwds)
        model.write(test_fname,
                    format="bar",
                    io_options=io_options)
        try:
            self.assertTrue(cmp(test_fname, baseline_fname))
        except:
            with open(test_fname, 'r') as f1, open(baseline_fname, 'r') as f2:
                f1_contents = f1.read().replace(' ;', ';').split()
                f2_contents = f2.read().replace(' ;', ';').split()
                self.assertEqual(f1_contents, f2_contents)
        self._cleanup(test_fname)

    def _gen_expression(self, terms):
        terms = list(terms)
        expr = 0.0
        for term in terms:
            if type(term) is tuple:
                prodterms = list(term)
                prodexpr = 1.0
                for x in prodterms:
                    prodexpr *= x
                expr += prodexpr
            else:
                expr += term
        return expr

    def test_no_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c,
                 (model.a, model.a), (model.b, model.b), (model.c, model.c),
                 (model.a, model.b), (model.a, model.c), (model.b, model.c)]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_no_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_no_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1,2], rule=lambda m, i: model.a == i)

        for key in components:
            model.add_component(key, components[key])

        self._check_baseline(model, file_determinism=2)

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)
        self._check_baseline(model)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)
        self._check_baseline(model)

    def test_var_on_nonblock(self):
        class Foo(Block().__class__):
            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype',Foo)
                super(Foo,self).__init__(*args, **kwds)

        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.deactivate()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)
        self._check_baseline(model)

    def test_trig_generates_exception(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,2*3.1415))
        m.obj = Objective(expr=sin(m.x))
        with self.assertRaisesRegex(
            RuntimeError,
            'The BARON .BAR format does not support the unary function "sin"'
            ):
            test_fname = self._get_fnames()[1]
            self._cleanup(test_fname)
            m.write(test_fname, format="bar")
        self._cleanup(test_fname)

    def test_exponential_NPV(self):
        m = ConcreteModel()
        m.x = Var()
        m.obj = Objective(expr=m.x**2)
        m.p = Param(initialize=1, mutable=True)
        m.c = Constraint(expr=m.x * m.p ** 1.2 == 0)
        self._check_baseline(m)

    def test_branching_priorities(self):
        m = ConcreteModel()
        m.x = Var(within=Binary)
        m.y = Var([1, 2], within=Binary)
        m.c = Constraint(expr=m.y[1]*m.y[2] - 2*m.x >= 0)
        m.obj = Objective(expr=m.y[1]+m.y[2], sense=maximize)
        m.priority = Suffix(direction=Suffix.EXPORT)
        m.priority[m.x] = 1
        m.priority[m.y] = 2
        self._check_baseline(m)

    def test_invalid_suffix(self):
        m = ConcreteModel()
        m.x = Var(within=Binary)
        m.y = Var([1, 2], within=Binary)
        m.c = Constraint(expr=m.y[1]*m.y[2] - 2*m.x >= 0)
        m.obj = Objective(expr=m.y[1]+m.y[2], sense=maximize)
        m.priorities = Suffix(direction=Suffix.EXPORT)
        m.priorities[m.x] = 1
        m.priorities[m.y] = 2
        with self.assertRaisesRegex(ValueError, "The BARON writer can not "
                                    "export suffix with name 'priorities'."):
            m.write(StringIO(), format='bar')


#class TestBaron_writer(unittest.TestCase):
class XTestBaron_writer(object):
    """These tests verified that the BARON writer complained loudly for
    variables that were not on the model, not on an active block, or not
    on a Block ctype.  As we are relaxing that requirement throughout
    Pyomo, these tests have been disabled."""

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix+".bar.baseline", prefix+".bar.out"

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            KeyError,
            model.write, test_fname, format='bar')
        self._cleanup(test_fname)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            KeyError,
            model.write, test_fname, format='bar' )
        self._cleanup(test_fname)

    def test_var_on_nonblock(self):
        class Foo(Block().__class__):
            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype',Foo)
                super(Foo,self).__init__(*args, **kwds)

        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            KeyError,
            model.write, test_fname, format='bar')
        self._cleanup(test_fname)


if __name__ == "__main__":
    unittest.main()
