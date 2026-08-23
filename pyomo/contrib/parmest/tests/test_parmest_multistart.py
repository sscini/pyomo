#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026 National Technology and Engineering Solutions of
#  Sandia, LLC Under the terms of Contract DE-NA0003525 with National
#  Technology and Engineering Solutions of Sandia, LLC, the U.S. Government
#  retains certain rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, pandas as pd
from unittest.mock import patch

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

ipopt_available = pyo.SolverFactory("ipopt").available()



