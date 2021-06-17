#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Script to generate the installer for pyomo.
"""

import os
import platform
import sys
from setuptools import setup, find_packages, Command
try:
    from setuptools import DistutilsOptionError
except ImportError:
    from distutils.errors import DistutilsOptionError

def read(*rnames):
    with open(os.path.join(os.path.dirname(__file__), *rnames)) as README:
        # Strip all leading badges up to, but not including the COIN-OR
        # badge so that they do not appear in the PyPI description
        while True:
            line = README.readline()
            if 'COIN-OR' in line:
                break
            if line.strip() and '[![' not in line:
                break
        return line + README.read()

def get_version():
    # Source pyomo/version/info.py to get the version number
    _verInfo = dict(globals())
    _verFile = os.path.join(os.path.dirname(__file__),
                            'pyomo','version','info.py')
    with open(_verFile) as _FILE:
        exec(_FILE.read(), _verInfo)
    return _verInfo['__version__']

CYTHON_REQUIRED = "required"
if not any(arg.startswith(cmd)
           for cmd in ('build','install','bdist') for arg  in sys.argv):
    using_cython = False
else:
    using_cython = "automatic"
if '--with-cython' in sys.argv:
    using_cython = CYTHON_REQUIRED
    sys.argv.remove('--with-cython')
if '--without-cython' in sys.argv:
    using_cython = False
    sys.argv.remove('--without-cython')

ext_modules = []
if using_cython:
    try:
        if platform.python_implementation() != "CPython":
            # break out of this try-except (disable Cython)
            raise RuntimeError("Cython is only supported under CPython")
        from Cython.Build import cythonize
        #
        # Note: The Cython developers recommend that you destribute C source
        # files to users.  But this is fine for evaluating the utility of Cython
        #
        import shutil
        files = [
            "pyomo/core/expr/numvalue.pyx",
            "pyomo/core/expr/numeric_expr.pyx",
            "pyomo/core/expr/logical_expr.pyx",
            #"pyomo/core/expr/visitor.pyx",
            "pyomo/core/util.pyx",
            "pyomo/repn/standard_repn.pyx",
            "pyomo/repn/plugins/cpxlp.pyx",
            "pyomo/repn/plugins/gams_writer.pyx",
            "pyomo/repn/plugins/baron_writer.pyx",
            "pyomo/repn/plugins/ampl/ampl_.pyx",
        ]
        for f in files:
            shutil.copyfile(f[:-1], f)
        ext_modules = cythonize(files, compiler_directives={
            "language_level": 3 if sys.version_info >= (3, ) else 2})
    except:
        if using_cython == CYTHON_REQUIRED:
            print("""
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
""")
            raise
        using_cython = False


class dependencies(Command):
    """Custom setuptools command

    This will output the list of dependencies (so we can easily pass
    them on to, e.g., conda)

    """
    description = "list the dependencies for this package"
    user_options = [
        ('extras=', None, 'extra targets to include'),
    ]

    def initialize_options(self):
        self.extras = None

    def finalize_options(self):
        if self.extras is not None:
            self.extras = [
                e for e in (_.strip() for _ in self.extras.split(',')) if e
            ]
            for e in self.extras:
                if e not in setup_kwargs['extras_require']:
                    raise DistutilsOptionError(
                        "extras can only include {%s}"
                        % (', '.join(setup_kwargs['extras_require'])))

    def run(self):
        deps = list(self._print_deps(setup_kwargs['install_requires']))
        if self.extras is not None:
            for e in self.extras:
                deps.extend(self._print_deps(setup_kwargs['extras_require'][e]))
        print(' '.join(deps))

    def _print_deps(self, deplist):
        implementation_name = sys.implementation.name
        python_version = '.'.join(platform.python_version_tuple()[:2])
        for entry in deplist:
            dep, _, condition = (_.strip() for _ in entry.partition(';'))
            if condition:
                if not eval(condition):
                    continue
            yield "'" + dep + "'"


setup_kwargs = dict(
    name = 'Pyomo',
    #
    # Note: the release number is set in pyomo/version/info.py
    #
    cmdclass = {'dependencies': dependencies},
    version = get_version(),
    maintainer = 'Pyomo Developer Team',
    maintainer_email = 'pyomo-developers@googlegroups.com',
    url = 'http://pyomo.org',
    license = 'BSD',
    platforms = ["any"],
    description = 'Pyomo: Python Optimization Modeling Objects',
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    keywords = ['optimization'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules' ],
    python_requires = '>=3.6',
    install_requires = [
        'ply',
    ],
    extras_require = {
        'tests': [
            'coverage',
            'nose',
            'parameterized',
            'pybind11',
        ],
        'docs': [
            'Sphinx>2',
            'sphinx-copybutton',
            'sphinx-rtd-theme>0.5',
            'sphinxcontrib-jsmath',
            'sphinxcontrib-napoleon',
            'numpy', # Needed by autodoc for pynumero
            'scipy', # Needed by autodoc for pynumero
        ],
        'optional': [
            'dill',      # No direct use, but improves lambda pickle
            'ipython',   # contrib.viewer
            'networkx',  # network, incidence_analysis, community_detection
            'openpyxl',  # dataportals
            #'pathos',   # requested for #963, but PR currently closed
            'pint',      # units
            #'pyro4',    # used by PySP; no longer needed by core Pyomo
            'python-louvain', # community_detection
            'pyyaml',    # core
            'sympy',     # differentiation
            'xlrd',      # dataportals
            'z3-solver', # community_detection
            # The following optional dependencies are difficult to
            # install on PyPy (due to the numpy dependency), so we
            # will only "require" them on other (CPython) platforms:
            'casadi; implementation_name!="pypy"', # dae
            'matplotlib; implementation_name!="pypy"',
            'numdifftools; implementation_name!="pypy"', # pynumero
            'numpy; implementation_name!="pypy"',
            'pandas; implementation_name!="pypy"',
            'scipy; implementation_name!="pypy"',
            'seaborn; implementation_name!="pypy"', # parmest.graphics
        ],
    },
    packages = find_packages(exclude=("scripts",)),
    package_data = {
        "pyomo.contrib.appsi.cmodel": ["src/*"],
        "pyomo.contrib.mcpp": ["*.cpp"],
        "pyomo.contrib.pynumero": ['src/*', 'src/tests/*'],
        "pyomo.contrib.viewer": ["*.ui"],
    },
    #include_package_data=True,
    ext_modules = ext_modules,
    entry_points = """
    [console_scripts]
    pyomo = pyomo.scripting.pyomo_main:main_console_script

    [pyomo.command]
    pyomo.help = pyomo.scripting.driver_help
    pyomo.viewer=pyomo.contrib.viewer.pyomo_viewer
    """
)


try:
    setup(**setup_kwargs)
except SystemExit as e_info:
    # Cython can generate a SystemExit exception on Windows if the
    # environment is missing / has an incorrect Microsoft compiler.
    # Since Cython is not strictly required, we will disable Cython and
    # try re-running setup(), but only for this very specific situation.
    if 'Microsoft Visual C++' not in str(e_info):
        raise
    elif using_cython == CYTHON_REQUIRED:
        print("""
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
""")
        raise
    else:
        print("""
ERROR: setup() failed:
    %s
Re-running setup() without the Cython modules
""" % (str(e_info),))
        setup_kwargs['ext_modules'] = []
        setup(**setup_kwargs)
        print("""
WARNING: Installation completed successfully, but the attempt to cythonize
         core Pyomo modules failed.  Cython provides performance
         optimizations and is not required for any Pyomo functionality.
         Cython returned the following error:
   "%s"
""" % (str(e_info),))
