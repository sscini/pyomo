#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.base.misc
#

import re
import sys
import subprocess

import pyomo.common.unittest as unittest


class ImportData(object):
    def __init__(self):
        self.tpl = {}
        self.pyomo = {}

    def update(self, other):
        self.tpl.update(other.tpl)
        self.pyomo.update(other.pyomo)


def collect_import_time(module):
    output = subprocess.check_output(
        [sys.executable, '-X', 'importtime', '-c', 'import %s' % (module,)],
        stderr=subprocess.STDOUT,
    )
    # Note: test only runs in PY3
    output = output.decode()
    line_re = re.compile(r'.*:\s*(\d+) \|\s*(\d+) \| ( *)([^ ]+)')
    data = []
    for line in output.splitlines():
        g = line_re.match(line)
        if not g:
            continue
        _self = int(g.group(1))
        _cumul = int(g.group(2))
        _level = len(g.group(3)) // 2
        _module = g.group(4)
        # print("%6d %8d %2d %s" % (_self, _cumul, _level, _module))
        while len(data) < _level + 1:
            data.append(ImportData())
        if len(data) > _level + 1:
            assert len(data) == _level + 2
            inner = data.pop()
            inner.tpl = {
                (k if '(from' in k else "%s (from %s)" % (k, _module), v)
                for k, v in inner.tpl.items()
            }
            if _module.startswith('pyomo'):
                data[_level].update(inner)
                data[_level].pyomo[_module] = _self
            else:
                if _level > 0:
                    data[_level].tpl[_module] = _cumul
        elif _module.startswith('pyomo'):
            data[_level].pyomo[_module] = _self
        elif _level > 0:
            data[_level].tpl[_module] = _self
    assert len(data) == 1
    return data[0]


class TestPyomoEnviron(unittest.TestCase):
    def test_not_auto_imported(self):
        rc = subprocess.call(
            [
                sys.executable,
                '-c',
                'import pyomo.core, sys; '
                'sys.exit( 1 if "pyomo.environ" in sys.modules else 0 )',
            ]
        )
        if rc:
            self.fail(
                "Importing pyomo.core automatically imports "
                "pyomo.environ and it should not."
            )

    @unittest.skipIf(
        'pypy_version_info' in dir(sys), "PyPy does not support '-X importtime"
    )
    def test_tpl_import_time(self):
        data = collect_import_time('pyomo.environ')
        pyomo_time = sum(data.pyomo.values())
        tpl_time = sum(data.tpl.values())
        total = float(pyomo_time + tpl_time)
        print("Pyomo (by module time):")
        print(
            "\n".join(
                "   %s: %s" % i for i in sorted(data.pyomo.items(), key=lambda x: x[1])
            )
        )
        print("TPLS:")
        _line_fmt = "   %%%ds: %%6d %%s" % (
            max(len(k[: k.find(' ')]) for k in data.tpl),
        )
        print(
            "\n".join(
                _line_fmt % (k[: k.find(' ')], v, k[k.find(' ') :])
                for k, v in sorted(data.tpl.items())
            )
        )
        tpl = {}
        for k, v in data.tpl.items():
            _mod = k[: k.find(' ')].split('.')[0]
            tpl[_mod] = tpl.get(_mod, 0) + v
        tpl_by_time = sorted(tpl.items(), key=lambda x: x[1])
        print("TPLS (by package time):")
        print(
            "\n".join(
                "   %12s: %6d (%4.1f%%)" % (m, t, 100 * t / total)
                for m, t in tpl_by_time
            )
        )
        print("Pyomo:    %6d (%4.1f%%)" % (pyomo_time, 100 * pyomo_time / total))
        print("TPL:      %6d (%4.1f%%)" % (tpl_time, 100 * tpl_time / total))
        # Arbitrarily choose a threshold 10% more than the expected
        # value (at time of writing, TPL imports were 52-57% of the
        # import time on a development machine)
        self.assertLess(tpl_time / total, 0.65)
        # Spot-check the (known) worst offenders.  The following are
        # modules from the "standard" library.  Their order in the list
        # of slow-loading TPLs can vary from platform to platform.
        ref = {
            '__future__',
            'argparse',
            'ast',  # Imported on Windows
            'backports_abc',  # Imported by cython on Linux
            'base64',  # Imported on Windows
            'bisect',  # Imported by dae, dataportal, contrib/mpc
            'cPickle',
            'csv',
            'ctypes',  # mandatory import in core/base/external.py; TODO: fix this
            'datetime',  # imported by contrib.solver
            'decimal',
            'gc',  # Imported on MacOS, Windows; Linux in 3.10
            'glob',
            'heapq',  # Added in Python 3.10
            'importlib',
            'inspect',
            'json',  # Imported on Windows
            'locale',  # Added in Python 3.9
            'logging',
            'pickle',
            'platform',
            'shlex',
            'socket',  # Imported on MacOS, Windows; Linux in 3.10
            'subprocess',
            'tempfile',  # Imported on MacOS, Windows
            'textwrap',
            'typing',
            'win32file',  # Imported on Windows
            'win32pipe',  # Imported on Windows
        }
        # Non-standard-library TPLs that Pyomo will load unconditionally
        ref.add('ply')
        diff = set(_[0] for _ in tpl_by_time[-5:]).difference(ref)
        self.assertEqual(
            diff, set(), "Unexpected module found in 5 slowest-loading TPL modules"
        )


if __name__ == "__main__":
    # Running this file as a script will print out the package timing
    # information from test_tpl_import_time()
    unittest.main()
