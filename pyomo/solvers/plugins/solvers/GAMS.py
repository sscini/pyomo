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

from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess

from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory

import pyomo.common
from pyomo.common.dependencies import pathlib
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream

from pyomo.opt.base.solvers import _extract_version

from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable

import pyomo.core.base.suffix
import pyomo.core.kernel.suffix

from pyomo.opt.results import (
    SolverResults,
    SolverStatus,
    Solution,
    SolutionStatus,
    TerminationCondition,
)

from pyomo.common.dependencies import attempt_import
import struct


def _gams_importer():
    try:
        import gams.core.gdx as gdx

        return gdx
    except ImportError:
        try:
            # fall back to the pre-GAMS-45.0 API
            import gdxcc

            return gdxcc
        except:
            # suppress the error from the old API and reraise the current API import error
            pass
        raise


gdxcc, gdxcc_available = attempt_import('gdxcc', importer=_gams_importer)

logger = logging.getLogger('pyomo.solvers')


class _GAMSSolver(object):
    """Aggregate of common methods for GAMS interfaces"""

    def __init__(self, **kwds):
        self._version = None
        self._default_variable_value = None
        self._metasolver = False

        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

        self.options = Bunch()

    def version(self):
        """Returns a 4-tuple describing the solver executable version."""
        if self._version is None:
            self._version = self._get_version()
        return self._version

    def warm_start_capable(self):
        """True is the solver can accept a warm-start solution."""
        return True

    def default_variable_value(self):
        return self._default_variable_value

    def set_options(self, istr):
        if isinstance(istr, str):
            istr = self._options_string_to_dict(istr)
        for key in istr:
            if not istr[key] is None:
                setattr(self.options, key, istr[key])

    @staticmethod
    def _options_string_to_dict(istr):
        ans = {}
        istr = istr.strip()
        if not istr:
            return ans
        if istr[0] == "'" or istr[0] == '"':
            istr = eval(istr)
        tokens = shlex.split(istr)
        for token in tokens:
            index = token.find('=')
            if index == -1:
                raise ValueError(
                    "Solver options must have the form option=value: '%s'" % istr
                )
            try:
                val = eval(token[(index + 1) :])
            except:
                val = token[(index + 1) :]
            ans[token[:index]] = val
        return ans

    def _simple_model(self, n):
        return """
            option limrow = 0;
            option limcol = 0;
            option solprint = off;
            set I / 1 * %s /;
            variables ans;
            positive variables x(I);
            equations obj;
            obj.. ans =g= sum(I, x(I));
            model test / all /;
            solve test using lp minimizing ans;
            """ % (
            n,
        )

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


@SolverFactory.register('gams', doc='The GAMS modeling language')
class GAMSSolver(_GAMSSolver):
    """
    A generic interface to GAMS solvers.

    Pass solver_io keyword arg to SolverFactory to choose solver mode:
        solver_io='direct' or 'python' to use GAMS Python API
            Requires installation, visit Python API page on gams.com for help.
        solver_io='shell' or 'gms' to use command line to call gams
            Requires the gams executable be on your system PATH.
    """

    def __new__(cls, *args, **kwds):
        mode = kwds.pop('solver_io', 'shell')
        if mode is None:
            mode = 'shell'

        if mode == 'direct' or mode == 'python':
            return SolverFactory('_gams_direct', **kwds)
        if mode == 'shell' or mode == 'gms':
            return SolverFactory('_gams_shell', **kwds)
        else:
            logger.error('Unknown IO type: %s' % mode)
            return


@SolverFactory.register(
    '_gams_direct', doc='Direct python interface to the GAMS modeling language'
)
class GAMSDirect(_GAMSSolver):
    """
    A generic python interface to GAMS solvers.

    Visit Python API page on gams.com for installation help.
    """

    def available(self, exception_flag=True):
        """True if the solver is available."""
        try:
            from gams import GamsWorkspace, DebugLevel
        except ImportError as e:
            if not exception_flag:
                return False
            raise ImportError(
                "Import of gams failed - GAMS direct "
                "solver functionality is not available.\n"
                "GAMS message: %s" % (e,)
            )
        avail = self._run_simple_model(1)
        if not avail and exception_flag:
            raise NameError(
                "'gams' command failed to solve a simple model - "
                "GAMS shell solver functionality is not available."
            )
        return avail

    def license_is_valid(self):
        # New versions of the community license can run LPs up to 5k
        return self._run_simple_model(5001)

    def _get_version(self):
        """Returns a tuple describing the solver executable version."""
        if not self.available(exception_flag=False):
            return _extract_version('')
        from gams import GamsWorkspace

        workspace = GamsWorkspace()
        version = tuple(int(i) for i in workspace._version.split('.')[:4])
        while len(version) < 4:
            version += (0,)
        return version

    def _run_simple_model(self, n):
        tmpdir = mkdtemp()
        try:
            from gams import GamsWorkspace, DebugLevel

            workspace = GamsWorkspace(debug=DebugLevel.Off, working_directory=tmpdir)
            t1 = workspace.add_job_from_string(self._simple_model(n))
            t1.run()
            return True
        except:
            return False
        finally:
            shutil.rmtree(tmpdir)

    def solve(self, *args, **kwds):
        """
        Solve a model via the GAMS Python API.

        Keyword Arguments
        -----------------
        tee=False: bool
            Output GAMS log to stdout.
        logfile=None: str
            Filename to output GAMS log to a file.
        load_solutions=True: bool
            Load solution into model. If False, the results
            object will contain the solution data.
        keepfiles=False: bool
            Keep temporary files. Equivalent of DebugLevel.KeepFiles.
            Summary of temp files can be found in _gams_py_gjo0.pf
        tmpdir=None: str
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
            By default uses the system default temporary path.
        report_timing=False: bool
            Print timing reports for presolve, solver, postsolve, etc.
        io_options: dict
            Options that get passed to the writer.
            See writer in pyomo.repn.plugins.gams_writer for details.
            Updated with any other keywords passed to solve method.
        """

        # Make sure available() doesn't crash
        self.available()

        from gams import GamsWorkspace, DebugLevel

        try:
            from gams import GamsExceptionExecution
        except ImportError:
            from gams.workspace import GamsExceptionExecution

        if len(args) != 1:
            raise ValueError(
                'Exactly one model must be passed to solve method of GAMSSolver.'
            )
        model = args[0]

        # self.options are default for each run, overwritten by kwds
        options = dict()
        options.update(self.options)
        options.update(kwds)

        load_solutions = options.pop("load_solutions", True)
        tee = options.pop("tee", False)
        logfile = options.pop("logfile", None)
        keepfiles = options.pop("keepfiles", False)
        tmpdir = options.pop("tmpdir", None)
        report_timing = options.pop("report_timing", False)
        io_options = options.pop("io_options", {})

        # Pass remaining keywords to writer, which will handle
        # any unrecognized arguments
        io_options.update(options)

        initial_time = time.time()

        # Because GAMS changes the CWD when running the solver, we need
        # to convert user-provided file names to absolute paths
        # (relative to the current directory)
        if logfile is not None:
            logfile = os.path.abspath(logfile)

        ####################################################################
        # Presolve
        ####################################################################

        # Create StringIO stream to pass to gams_writer, on which the
        # model file will be written. The writer also passes this StringIO
        # back, but output_file is defined in advance for clarity.
        output_file = StringIO()
        if isinstance(model, IBlock):
            # Kernel blocks have slightly different write method
            smap_id = model.write(
                filename=output_file,
                format=ProblemFormat.gams,
                _called_by_solver=True,
                **io_options,
            )
            symbolMap = getattr(model, "._symbol_maps")[smap_id]
        else:
            (_, smap_id) = model.write(
                filename=output_file, format=ProblemFormat.gams, io_options=io_options
            )
            symbolMap = model.solutions.symbol_map[smap_id]

        presolve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for presolve"
                % (presolve_completion_time - initial_time)
            )

        ####################################################################
        # Apply solver
        ####################################################################

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keepfiles. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = True
        if tmpdir is not None and os.path.exists(tmpdir):
            newdir = False

        workspace = GamsWorkspace(
            debug=DebugLevel.KeepFiles if keepfiles else DebugLevel.Off,
            working_directory=tmpdir,
        )

        t1 = workspace.add_job_from_string(output_file.getvalue())

        try:
            with OutputStream(tee=tee, logfile=logfile) as output_stream:
                t1.run(output=output_stream)
        except GamsExceptionExecution as e:
            try:
                if e.rc == 3:
                    # Execution Error
                    check_expr_evaluation(model, symbolMap, 'direct')
            finally:
                # Always name working directory or delete files,
                # regardless of any errors.
                if keepfiles:
                    print(
                        "\nGAMS WORKING DIRECTORY: %s\n" % workspace.working_directory
                    )
                elif tmpdir is not None:
                    # Garbage collect all references to t1.out_db
                    # So that .gdx file can be deleted
                    t1 = rec = rec_lo = rec_hi = None
                    file_removal_gams_direct(tmpdir, newdir)
                raise
        except:
            # Catch other errors and remove files first
            if keepfiles:
                print("\nGAMS WORKING DIRECTORY: %s\n" % workspace.working_directory)
            elif tmpdir is not None:
                # Garbage collect all references to t1.out_db
                # So that .gdx file can be deleted
                t1 = rec = rec_lo = rec_hi = None
                file_removal_gams_direct(tmpdir, newdir)
            raise

        solve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for solver"
                % (solve_completion_time - presolve_completion_time)
            )

        ####################################################################
        # Postsolve
        ####################################################################

        # import suffixes must be on the top-level model
        if isinstance(model, IBlock):
            model_suffixes = list(
                comp.storage_key
                for comp in pyomo.core.kernel.suffix.import_suffix_generator(
                    model, active=True, descend_into=False
                )
            )
        else:
            model_suffixes = list(
                name
                for (
                    name,
                    comp,
                ) in pyomo.core.base.suffix.active_import_suffix_generator(model)
            )
        extract_dual = 'dual' in model_suffixes
        extract_rc = 'rc' in model_suffixes

        results = SolverResults()
        results.problem.name = os.path.join(
            workspace.working_directory, t1.name + '.gms'
        )
        results.problem.lower_bound = t1.out_db["OBJEST"].find_record().value
        results.problem.upper_bound = t1.out_db["OBJEST"].find_record().value
        results.problem.number_of_variables = t1.out_db["NUMVAR"].find_record().value
        results.problem.number_of_constraints = t1.out_db["NUMEQU"].find_record().value
        results.problem.number_of_nonzeros = t1.out_db["NUMNZ"].find_record().value
        results.problem.number_of_binary_variables = None
        # Includes binary vars:
        results.problem.number_of_integer_variables = (
            t1.out_db["NUMDVAR"].find_record().value
        )
        results.problem.number_of_continuous_variables = (
            t1.out_db["NUMVAR"].find_record().value
            - t1.out_db["NUMDVAR"].find_record().value
        )
        results.problem.number_of_objectives = 1  # required by GAMS writer
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'Only one objective is allowed.'
        obj = obj[0]
        objctvval = t1.out_db["OBJVAL"].find_record().value
        results.problem.sense = obj.sense
        if obj.is_minimizing():
            results.problem.upper_bound = objctvval
        else:
            results.problem.lower_bound = objctvval

        results.solver.name = "GAMS " + str(self.version())

        # Init termination condition to None to give preference to this first
        # block of code, only set certain TC's below if it's still None
        results.solver.termination_condition = None
        results.solver.message = None

        solvestat = t1.out_db["SOLVESTAT"].find_record().value
        if solvestat == 1:
            results.solver.status = SolverStatus.ok
        elif solvestat == 2:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif solvestat == 3:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif solvestat == 5:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif solvestat == 7:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = (
                TerminationCondition.licensingProblems
            )
        elif solvestat == 8:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
        elif solvestat == 10:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.solverFailure
        elif solvestat == 11:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = (
                TerminationCondition.internalSolverError
            )
        elif solvestat == 4:
            results.solver.status = SolverStatus.warning
            results.solver.message = "Solver quit with a problem (see LST file)"
        elif solvestat in (9, 12, 13):
            results.solver.status = SolverStatus.error
        elif solvestat == 6:
            results.solver.status = SolverStatus.unknown

        results.solver.return_code = 0
        # Not sure if this value is actually user time
        # "the elapsed time it took to execute a solve statement in total"
        results.solver.user_time = t1.out_db["ETSOLVE"].find_record().value
        results.solver.system_time = None
        results.solver.wallclock_time = None
        results.solver.termination_message = None

        soln = Solution()

        modelstat = t1.out_db["MODELSTAT"].find_record().value
        if modelstat == 1:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 2:
            results.solver.termination_condition = TerminationCondition.locallyOptimal
            soln.status = SolutionStatus.locallyOptimal
        elif modelstat in [3, 18]:
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif modelstat in [4, 5, 6, 10, 19]:
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif modelstat == 7:
            results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        elif modelstat == 8:
            # 'Integer solution model found'
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 9:
            results.solver.termination_condition = (
                TerminationCondition.intermediateNonInteger
            )
            soln.status = SolutionStatus.other
        elif modelstat == 11:
            # Should be handled above, if modelstat and solvestat both
            # indicate a licensing problem
            if results.solver.termination_condition is None:
                results.solver.termination_condition = (
                    TerminationCondition.licensingProblems
                )
            soln.status = SolutionStatus.error
        elif modelstat in [12, 13]:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif modelstat == 14:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.noSolution
            soln.status = SolutionStatus.unknown
        elif modelstat in [15, 16, 17]:
            # Having to do with CNS models,
            # not sure what to make of status descriptions
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.unsure
        else:
            # This is just a backup catch, all cases are handled above
            soln.status = SolutionStatus.error

        soln.gap = abs(results.problem.upper_bound - results.problem.lower_bound)

        for sym, obj in symbolMap.bySymbol.items():
            if isinstance(model, IBlock):
                # Kernel variables have no 'parent_component'
                if obj.ctype is IObjective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.ctype is not IVariable:
                    continue
            else:
                if obj.parent_component().ctype is Objective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.parent_component().ctype is not Var:
                    continue
            rec = t1.out_db[sym].find_record()
            # obj.value = rec.level
            soln.variable[sym] = {"Value": rec.level}
            if extract_rc and not math.isnan(rec.marginal):
                # Do not set marginals to nan
                # model.rc[obj] = rec.marginal
                soln.variable[sym]['rc'] = rec.marginal

        if extract_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if c.body.is_fixed() or (not (c.has_lb() or c.has_ub())):
                    # the constraint was not sent to GAMS
                    continue
                sym = symbolMap.getSymbol(c)
                if c.equality:
                    rec = t1.out_db[sym].find_record()
                    if not math.isnan(rec.marginal):
                        # model.dual[c] = rec.marginal
                        soln.constraint[sym] = {'dual': rec.marginal}
                    else:
                        # Solver didn't provide marginals,
                        # nothing else to do here
                        break
                else:
                    # Inequality, assume if 2-sided that only
                    # one side's marginal is nonzero
                    # Negate marginal for _lo equations
                    marg = 0
                    if c.lower is not None:
                        rec_lo = t1.out_db[sym + '_lo'].find_record()
                        marg -= rec_lo.marginal
                    if c.upper is not None:
                        rec_hi = t1.out_db[sym + '_hi'].find_record()
                        marg += rec_hi.marginal
                    if not math.isnan(marg):
                        # model.dual[c] = marg
                        soln.constraint[sym] = {'dual': marg}
                    else:
                        # Solver didn't provide marginals,
                        # nothing else to do here
                        break

        results.solution.insert(soln)

        if keepfiles:
            print("\nGAMS WORKING DIRECTORY: %s\n" % workspace.working_directory)
        elif tmpdir is not None:
            # Garbage collect all references to t1.out_db
            # So that .gdx file can be deleted
            t1 = rec = rec_lo = rec_hi = None
            file_removal_gams_direct(tmpdir, newdir)

        ####################################################################
        # Finish with results
        ####################################################################

        results._smap_id = smap_id
        results._smap = None
        if isinstance(model, IBlock):
            if len(results.solution) == 1:
                results.solution(0).symbol_map = getattr(model, "._symbol_maps")[
                    results._smap_id
                ]
                results.solution(0).default_variable_value = (
                    self._default_variable_value
                )
                if load_solutions:
                    model.load_solution(results.solution(0))
            else:
                assert len(results.solution) == 0
            # see the hack in the write method
            # we don't want this to stick around on the model
            # after the solve
            assert len(getattr(model, "._symbol_maps")) == 1
            delattr(model, "._symbol_maps")
            del results._smap_id
            if load_solutions and (len(results.solution) == 0):
                logger.error("No solution is available")
        else:
            if load_solutions:
                model.solutions.load_from(results)
                results._smap_id = None
                results.solution.clear()
            else:
                results._smap = model.solutions.symbol_map[smap_id]
                model.solutions.delete_symbol_map(smap_id)

        postsolve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for postsolve"
                % (postsolve_completion_time - solve_completion_time)
            )
            print(
                "      %6.2f seconds required total"
                % (postsolve_completion_time - initial_time)
            )

        return results


@SolverFactory.register(
    '_gams_shell', doc='Shell interface to the GAMS modeling language'
)
class GAMSShell(_GAMSSolver):
    """A generic shell interface to GAMS solvers."""

    def available(self, exception_flag=True):
        """True if the solver is available."""
        exe = pyomo.common.Executable("gams")
        if not exe.available():
            if not exception_flag:
                return False
            raise NameError(
                "No 'gams' command found on system PATH - GAMS shell "
                "solver functionality is not available."
            )
        # New versions of GAMS require a license to run anything.
        # Instead of parsing the output, we will try solving a trivial
        # model.
        avail = self._run_simple_model(1)
        if not avail and exception_flag:
            raise NameError(
                "'gams' command failed to solve a simple model - "
                "GAMS shell solver functionality is not available."
            )
        return avail

    def license_is_valid(self):
        # New versions of the community license can run LPs up to 5k
        return self._run_simple_model(5001)

    def _run_simple_model(self, n):
        solver_exec = self.executable()
        if solver_exec is None:
            return False
        tmpdir = mkdtemp()
        try:
            test = os.path.join(tmpdir, 'test.gms')
            with open(test, 'w') as FILE:
                FILE.write(self._simple_model(n))
            result = subprocess.run(
                [solver_exec, test, "curdir=" + tmpdir, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return not result.returncode
        finally:
            shutil.rmtree(tmpdir)
        return False

    def _default_executable(self):
        executable = pyomo.common.Executable("gams")
        if not executable:
            logger.warning(
                "Could not locate the 'gams' executable, "
                "which is required for solver gams"
            )
            self.enable = False
            return None
        return executable.path()

    def executable(self):
        """Returns the executable used by this solver."""
        return self._default_executable()

    def _get_version(self):
        """Returns a tuple describing the solver executable version."""
        solver_exec = self.executable()

        if solver_exec is None:
            return _extract_version('')
        else:
            # specify logging to stdout for windows compatibility
            cmd = [solver_exec, "audit", "lo=3"]
            results = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            return _extract_version(results.stdout)

    def _rewrite_path_win8p3(self, path):
        """
        Return the 8.3 short path on Windows; unchanged elsewhere.

        This change is in response to Pyomo/pyomo#3579 which reported
        that GAMS (direct) fails on Windows if there is a space in
        the path. This utility converts paths to their 8.3 short-path version
        (which never have spaces).
        """
        if not sys.platform.startswith("win"):
            return str(path)

        import ctypes, ctypes.wintypes as wt

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wt.LPCWSTR, wt.LPWSTR, wt.DWORD]

        # the file must exist, or Windows will not create a short name
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).touch(exist_ok=True)

        buf = ctypes.create_unicode_buffer(260)
        if GetShortPathNameW(str(path), buf, 260):
            return buf.value
        return str(path)

    def solve(self, *args, **kwds):
        """
        Solve a model via the GAMS executable.

        Keyword Arguments
        -----------------
        tee=False: bool
            Output GAMS log to stdout.
        logfile=None: str
            Filename to output GAMS log to a file.
        load_solutions=True: bool
            Load solution into model. If False, the results
            object will contain the solution data.
        keepfiles=False: bool
            Keep temporary files.
        tmpdir=None: str
            Specify directory path for storing temporary files.
            A directory will be created if one of this name doesn't exist.
            By default uses the system default temporary path.
        report_timing=False: bool
            Print timing reports for presolve, solver, postsolve, etc.
        io_options: dict
            Options that get passed to the writer.
            See writer in pyomo.repn.plugins.gams_writer for details.
            Updated with any other keywords passed to solve method.
            Note: put_results is not available for modification on
            GAMSShell solver.
        """

        # Make sure available() doesn't crash
        self.available()

        if len(args) != 1:
            raise ValueError(
                'Exactly one model must be passed to solve method of GAMSSolver.'
            )
        model = args[0]

        # self.options are default for each run, overwritten by kwds
        options = dict()
        options.update(self.options)
        options.update(kwds)

        load_solutions = options.pop("load_solutions", True)
        tee = options.pop("tee", False)
        logfile = options.pop("logfile", None)
        keepfiles = options.pop("keepfiles", False)
        tmpdir = options.pop("tmpdir", None)
        report_timing = options.pop("report_timing", False)
        io_options = options.pop("io_options", {})

        io_options.update(options)

        # Pass remaining keywords to writer, which will handle
        # any unrecognized arguments
        initial_time = time.time()

        # Because GAMS changes the CWD when running the solver, we need
        # to convert user-provided file names to absolute paths
        # (relative to the current directory)
        if logfile is not None:
            logfile = os.path.abspath(logfile)

        ####################################################################
        # Presolve
        ####################################################################

        # IMPORTANT - only delete the whole tmpdir if the solver was the one
        # that made the directory. Otherwise, just delete the files the solver
        # made, if not keepfiles. That way the user can select a directory
        # they already have, like the current directory, without having to
        # worry about the rest of the contents of that directory being deleted.
        newdir = False
        if tmpdir is None:
            tmpdir = mkdtemp()
            newdir = True
        elif not os.path.exists(tmpdir):
            # makedirs creates all necessary intermediate directories in order
            # to create the path to tmpdir, if they don't already exist.
            # However, if keepfiles is False, we only delete the final folder,
            # leaving the rest of the intermediate ones.
            os.makedirs(tmpdir)
            newdir = True

        output = "model.gms"
        output_filename = os.path.join(tmpdir, output)
        lst = "output.lst"
        lst_filename = os.path.join(tmpdir, lst)

        put_results = "results"
        io_options["put_results"] = put_results
        io_options.setdefault("put_results_format", 'gdx' if gdxcc_available else 'dat')

        if io_options['put_results_format'] == 'gdx':
            results_filename = os.path.join(tmpdir, "GAMS_MODEL_p.gdx")
            statresults_filename = os.path.join(tmpdir, "%s_s.gdx" % (put_results,))
        else:
            results_filename = os.path.join(tmpdir, "%s.dat" % (put_results,))
            statresults_filename = os.path.join(tmpdir, "%sstat.dat" % (put_results,))

        if isinstance(model, IBlock):
            # Kernel blocks have slightly different write method
            smap_id = model.write(
                filename=output_filename,
                format=ProblemFormat.gams,
                _called_by_solver=True,
                **io_options,
            )
            symbolMap = getattr(model, "._symbol_maps")[smap_id]
        else:
            (_, smap_id) = model.write(
                filename=output_filename,
                format=ProblemFormat.gams,
                io_options=io_options,
            )
            symbolMap = model.solutions.symbol_map[smap_id]

        presolve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for presolve"
                % (presolve_completion_time - initial_time)
            )

        ####################################################################
        # Apply solver
        ####################################################################

        exe = self.executable()
        command = [exe, output, "o=" + lst, "curdir=" + tmpdir]
        if tee and not logfile:
            # default behaviour of gams is to print to console, for
            # compatibility with windows and *nix we want to explicitly log to
            # stdout (see https://www.gams.com/latest/docs/UG_GamsCall.html)
            command.append("lo=3")
        elif not tee and not logfile:
            command.append("lo=0")
        elif not tee and logfile:
            command.append("lo=2")
        elif tee and logfile:
            command.append("lo=4")
        if logfile:
            command.append(f"lf={self._rewrite_path_win8p3(logfile)}")

        try:
            ostreams = [StringIO()]
            if tee:
                ostreams.append(sys.stdout)
            with TeeStream(*ostreams) as t:
                result = subprocess.run(command, stdout=t.STDOUT, stderr=t.STDERR)
            rc = result.returncode
            txt = ostreams[0].getvalue()

            if keepfiles:
                print("\nGAMS WORKING DIRECTORY: %s\n" % tmpdir)

            if rc == 1 or rc == 127:
                raise IOError("Command 'gams' was not recognized")
            elif rc != 0:
                if rc == 3:
                    # Execution Error
                    # Run check_expr_evaluation, which errors if necessary
                    check_expr_evaluation(model, symbolMap, 'shell')
                # If nothing was raised, or for all other cases, raise this
                logger.error(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )
                logger.error(txt)
                if os.path.exists(lst_filename):
                    with open(lst_filename, 'r') as FILE:
                        logger.error("GAMS Listing file:\n\n%s" % (FILE.read(),))
                raise RuntimeError(
                    "GAMS encountered an error during solve. "
                    "Check listing file for details."
                )

            if io_options['put_results_format'] == 'gdx':
                model_soln, stat_vars = self._parse_gdx_results(
                    results_filename, statresults_filename
                )
            else:
                model_soln, stat_vars = self._parse_dat_results(
                    results_filename, statresults_filename
                )
        finally:
            if not keepfiles:
                if newdir:
                    shutil.rmtree(tmpdir)
                else:
                    os.remove(output_filename)
                    os.remove(lst_filename)
                    os.remove(results_filename)
                    os.remove(statresults_filename)

        solve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for solver"
                % (solve_completion_time - presolve_completion_time)
            )

        ####################################################################
        # Postsolve
        ####################################################################

        # import suffixes must be on the top-level model
        if isinstance(model, IBlock):
            model_suffixes = list(
                comp.storage_key
                for comp in pyomo.core.kernel.suffix.import_suffix_generator(
                    model, active=True, descend_into=False
                )
            )
        else:
            model_suffixes = list(
                name
                for (
                    name,
                    comp,
                ) in pyomo.core.base.suffix.active_import_suffix_generator(model)
            )
        extract_dual = 'dual' in model_suffixes
        extract_rc = 'rc' in model_suffixes

        results = SolverResults()
        results.problem.name = output_filename
        results.problem.lower_bound = stat_vars["OBJEST"]
        results.problem.upper_bound = stat_vars["OBJEST"]
        results.problem.number_of_variables = stat_vars["NUMVAR"]
        results.problem.number_of_constraints = stat_vars["NUMEQU"]
        results.problem.number_of_nonzeros = stat_vars["NUMNZ"]
        results.problem.number_of_binary_variables = None
        # Includes binary vars:
        results.problem.number_of_integer_variables = stat_vars["NUMDVAR"]
        results.problem.number_of_continuous_variables = (
            stat_vars["NUMVAR"] - stat_vars["NUMDVAR"]
        )
        results.problem.number_of_objectives = 1  # required by GAMS writer
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, 'Only one objective is allowed.'
        obj = obj[0]
        objctvval = stat_vars["OBJVAL"]
        results.problem.sense = obj.sense
        if obj.is_minimizing():
            results.problem.upper_bound = objctvval
        else:
            results.problem.lower_bound = objctvval

        results.solver.name = "GAMS " + str(self.version())

        # Init termination condition to None to give preference to this first
        # block of code, only set certain TC's below if it's still None
        results.solver.termination_condition = None
        results.solver.message = None

        solvestat = stat_vars["SOLVESTAT"]
        if solvestat == 1:
            results.solver.status = SolverStatus.ok
        elif solvestat == 2:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif solvestat == 3:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif solvestat == 5:
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif solvestat == 7:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = (
                TerminationCondition.licensingProblems
            )
        elif solvestat == 8:
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
        elif solvestat == 10:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.solverFailure
        elif solvestat == 11:
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = (
                TerminationCondition.internalSolverError
            )
        elif solvestat == 4:
            results.solver.status = SolverStatus.warning
            results.solver.message = "Solver quit with a problem (see LST file)"
        elif solvestat in (9, 12, 13):
            results.solver.status = SolverStatus.error
        elif solvestat == 6:
            results.solver.status = SolverStatus.unknown

        results.solver.return_code = rc  # 0
        # Not sure if this value is actually user time
        # "the elapsed time it took to execute a solve statement in total"
        results.solver.user_time = stat_vars["ETSOLVE"]
        results.solver.system_time = None
        results.solver.wallclock_time = None
        results.solver.termination_message = None

        soln = Solution()

        modelstat = stat_vars["MODELSTAT"]
        if modelstat == 1:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 2:
            results.solver.termination_condition = TerminationCondition.locallyOptimal
            soln.status = SolutionStatus.locallyOptimal
        elif modelstat in [3, 18]:
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif modelstat in [4, 5, 6, 10, 19]:
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif modelstat == 7:
            results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        elif modelstat == 8:
            # 'Integer solution model found'
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif modelstat == 9:
            results.solver.termination_condition = (
                TerminationCondition.intermediateNonInteger
            )
            soln.status = SolutionStatus.other
        elif modelstat == 11:
            # Should be handled above, if modelstat and solvestat both
            # indicate a licensing problem
            if results.solver.termination_condition is None:
                results.solver.termination_condition = (
                    TerminationCondition.licensingProblems
                )
            soln.status = SolutionStatus.error
        elif modelstat in [12, 13]:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif modelstat == 14:
            if results.solver.termination_condition is None:
                results.solver.termination_condition = TerminationCondition.noSolution
            soln.status = SolutionStatus.unknown
        elif modelstat in [15, 16, 17]:
            # Having to do with CNS models,
            # not sure what to make of status descriptions
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.unsure
        else:
            # This is just a backup catch, all cases are handled above
            soln.status = SolutionStatus.error

        soln.gap = abs(results.problem.upper_bound - results.problem.lower_bound)

        has_rc_info = True
        for sym, obj in symbolMap.bySymbol.items():
            if isinstance(model, IBlock):
                # Kernel variables have no 'parent_component'
                if obj.ctype is IObjective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.ctype is not IVariable:
                    continue
            else:
                if obj.parent_component().ctype is Objective:
                    soln.objective[sym] = {'Value': objctvval}
                if obj.parent_component().ctype is not Var:
                    continue
            try:
                rec = model_soln[sym]
            except KeyError:
                # no solution returned
                rec = (float('nan'), float('nan'))
            # obj.value = float(rec[0])
            soln.variable[sym] = {"Value": float(rec[0])}
            if extract_rc and has_rc_info:
                try:
                    # model.rc[obj] = float(rec[1])
                    soln.variable[sym]['rc'] = float(rec[1])
                except ValueError:
                    # Solver didn't provide marginals
                    has_rc_info = False

        if extract_dual:
            for c in model.component_data_objects(Constraint, active=True):
                if (c.body.is_fixed()) or (not (c.has_lb() or c.has_ub())):
                    # the constraint was not sent to GAMS
                    continue
                sym = symbolMap.getSymbol(c)
                if c.equality:
                    try:
                        rec = model_soln[sym]
                    except KeyError:
                        # no solution returned
                        rec = (float('nan'), float('nan'))
                    try:
                        # model.dual[c] = float(rec[1])
                        soln.constraint[sym] = {'dual': float(rec[1])}
                    except ValueError:
                        # Solver didn't provide marginals
                        # nothing else to do here
                        break
                else:
                    # Inequality, assume if 2-sided that only
                    # one side's marginal is nonzero
                    # Negate marginal for _lo equations
                    marg = 0
                    if c.lower is not None:
                        try:
                            rec_lo = model_soln[sym + '_lo']
                        except KeyError:
                            # no solution returned
                            rec_lo = (float('nan'), float('nan'))
                        try:
                            marg -= float(rec_lo[1])
                        except ValueError:
                            # Solver didn't provide marginals
                            marg = float('nan')
                    if c.upper is not None:
                        try:
                            rec_hi = model_soln[sym + '_hi']
                        except KeyError:
                            # no solution returned
                            rec_hi = (float('nan'), float('nan'))
                        try:
                            marg += float(rec_hi[1])
                        except ValueError:
                            # Solver didn't provide marginals
                            marg = float('nan')
                    if not math.isnan(marg):
                        # model.dual[c] = marg
                        soln.constraint[sym] = {'dual': marg}
                    else:
                        # Solver didn't provide marginals
                        # nothing else to do here
                        break

        results.solution.insert(soln)

        ####################################################################
        # Finish with results
        ####################################################################

        results._smap_id = smap_id
        results._smap = None
        if isinstance(model, IBlock):
            if len(results.solution) == 1:
                results.solution(0).symbol_map = getattr(model, "._symbol_maps")[
                    results._smap_id
                ]
                results.solution(0).default_variable_value = (
                    self._default_variable_value
                )
                if load_solutions:
                    model.load_solution(results.solution(0))
            else:
                assert len(results.solution) == 0
            # see the hack in the write method
            # we don't want this to stick around on the model
            # after the solve
            assert len(getattr(model, "._symbol_maps")) == 1
            delattr(model, "._symbol_maps")
            del results._smap_id
            if load_solutions and (len(results.solution) == 0):
                logger.error("No solution is available")
        else:
            if load_solutions:
                model.solutions.load_from(results)
                results._smap_id = None
                results.solution.clear()
            else:
                results._smap = model.solutions.symbol_map[smap_id]
                model.solutions.delete_symbol_map(smap_id)

        postsolve_completion_time = time.time()
        if report_timing:
            print(
                "      %6.2f seconds required for postsolve"
                % (postsolve_completion_time - solve_completion_time)
            )
            print(
                "      %6.2f seconds required total"
                % (postsolve_completion_time - initial_time)
            )

        return results

    def _parse_gdx_results(self, results_filename, statresults_filename):
        model_soln = dict()
        stat_vars = dict.fromkeys(
            [
                'MODELSTAT',
                'SOLVESTAT',
                'OBJEST',
                'OBJVAL',
                'NUMVAR',
                'NUMEQU',
                'NUMDVAR',
                'NUMNZ',
                'ETSOLVE',
            ]
        )

        pgdx = gdxcc.new_gdxHandle_tp()
        ret = gdxcc.gdxCreateD(pgdx, os.path.dirname(self.executable()), 128)
        if not ret[0]:
            raise RuntimeError("GAMS GDX failure (gdxCreate): %s." % ret[1])

        if os.path.exists(statresults_filename):
            ret = gdxcc.gdxOpenRead(pgdx, statresults_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])

            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)

            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(
                ">d", bytes.fromhex("fffffffffffffffe")
            )[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)

            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                stat = ret[1]
                if not stat in stat_vars:
                    continue

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) == 0:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")

                if stat in ('OBJEST', 'OBJVAL', 'ETSOLVE'):
                    stat_vars[stat] = ret[2][0]
                else:
                    stat_vars[stat] = int(ret[2][0])

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        if os.path.exists(results_filename):
            ret = gdxcc.gdxOpenRead(pgdx, results_filename)
            if not ret[0]:
                raise RuntimeError("GAMS GDX failure (gdxOpenRead): %d." % ret[1])

            specVals = gdxcc.doubleArray(gdxcc.GMS_SVIDX_MAX)
            rc = gdxcc.gdxGetSpecialValues(pgdx, specVals)

            specVals[gdxcc.GMS_SVIDX_EPS] = sys.float_info.min
            specVals[gdxcc.GMS_SVIDX_UNDEF] = float("nan")
            specVals[gdxcc.GMS_SVIDX_PINF] = float("inf")
            specVals[gdxcc.GMS_SVIDX_MINF] = float("-inf")
            specVals[gdxcc.GMS_SVIDX_NA] = struct.unpack(
                ">d", bytes.fromhex("fffffffffffffffe")
            )[0]
            gdxcc.gdxSetSpecialValues(pgdx, specVals)

            i = 0
            while True:
                i += 1
                ret = gdxcc.gdxDataReadRawStart(pgdx, i)
                if not ret[0]:
                    break

                ret = gdxcc.gdxDataReadRaw(pgdx)
                if not ret[0] or len(ret[2]) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxDataReadRaw).")
                level = ret[2][0]
                dual = ret[2][1]

                ret = gdxcc.gdxSymbolInfo(pgdx, i)
                if not ret[0]:
                    break
                if len(ret) < 2:
                    raise RuntimeError("GAMS GDX failure (gdxSymbolInfo).")
                model_soln[ret[1]] = (level, dual)

            gdxcc.gdxDataReadDone(pgdx)
            gdxcc.gdxClose(pgdx)

        gdxcc.gdxFree(pgdx)
        gdxcc.gdxLibraryUnload()
        return model_soln, stat_vars

    def _parse_dat_results(self, results_filename, statresults_filename):
        with open(statresults_filename, 'r') as statresults_file:
            statresults_text = statresults_file.read()

        stat_vars = dict()
        # Skip first line of explanatory text
        for line in statresults_text.splitlines()[1:]:
            items = line.split()
            try:
                stat_vars[items[0]] = float(items[1])
            except ValueError:
                # GAMS printed NA, just make it nan
                stat_vars[items[0]] = float('nan')

        with open(results_filename, 'r') as results_file:
            results_text = results_file.read()

        model_soln = dict()
        # Skip first line of explanatory text
        for line in results_text.splitlines()[1:]:
            items = line.split()
            model_soln[items[0]] = (items[1], items[2])

        return model_soln, stat_vars


class OutputStream:
    """Output stream object for simultaneously writing to multiple streams.

    tee=False:
        If set writing to this stream will write to stdout.
    logfile=None:
        Optionally a logfile can be written.

    """

    def __init__(self, tee=False, logfile=None):
        """Initialize output stream object."""
        if tee:
            self.tee = sys.stdout
        else:
            self.tee = None
        self.logfile = logfile
        self.logfile_buffer = None

    def __enter__(self):
        """Enter context of output stream and open logfile if given."""
        if self.logfile is not None:
            self.logfile_buffer = open(self.logfile, 'a')
        return self

    def __exit__(self, *args, **kwargs):
        """Enter context of output stream and close logfile if necessary."""
        if self.logfile_buffer is not None:
            self.logfile_buffer.close()
        self.logfile_buffer = None

    def write(self, message):
        """Write messages to all streams."""
        if self.tee is not None:
            self.tee.write(message)
        if self.logfile_buffer is not None:
            self.logfile_buffer.write(message)

    def flush(self):
        """Needed for python3 compatibility."""
        if self.tee is not None:
            self.tee.flush()
        if self.logfile_buffer is not None:
            self.logfile_buffer.flush()


def check_expr_evaluation(model, symbolMap, solver_io):
    try:
        # Temporarily initialize uninitialized variables in order to call
        # value() on each expression to check domain violations
        uninit_vars = list()
        for var in model.component_data_objects(Var):
            if var.value is None:
                uninit_vars.append(var)
                var.set_value(0, skip_validation=True)

        # Constraints
        for con in model.component_data_objects(Constraint, active=True):
            if con.body.is_fixed():
                continue
            check_expr(con.body, con.name, solver_io)

        # Objective
        obj = list(model.component_data_objects(Objective, active=True))
        assert len(obj) == 1, "GAMS writer can only take 1 active objective"
        obj = obj[0]
        check_expr(obj.expr, obj.name, solver_io)
    finally:
        # Return uninitialized variables to None
        for var in uninit_vars:
            var.set_value(None)


def check_expr(expr, name, solver_io):
    # Check if GAMS will encounter domain violations in presolver
    # operations at current values, which are None (0) by default
    # Used to handle log and log10 violations, for example
    try:
        value(expr)
    except (ValueError, ZeroDivisionError):
        logger.warning(
            "While evaluating model.%s's expression, GAMS solver "
            "encountered an error.\nGAMS requires that all "
            "equations and expressions evaluate at initial values.\n"
            "Ensure variable values do not violate any domains, "
            "and use the warmstart=True keyword to solve()." % name
        )
        if solver_io == 'shell':
            # For shell, there is no previous exception to worry about
            # overwriting, so raise the ValueError.
            # But for direct, the GamsExceptionExecution will be raised.
            raise


def file_removal_gams_direct(tmpdir, newdir):
    if newdir:
        shutil.rmtree(tmpdir)
    else:
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.gms'))
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.lst'))
        os.remove(os.path.join(tmpdir, '_gams_py_gdb0.gdx'))
        # .pf file is not made when DebugLevel is Off
