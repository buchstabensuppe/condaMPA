import pydstool as dst
import numpy as np
from matplotlib import pyplot as plt
# we must give a name

DSargs = dst.args(name='Calcium channel model')
# parameters
DSargs.pars = { 'vl': -60,
               'vca': 120,
                 'i': 0,
                'gl': 2,
               'gca': 4,
                 'c': 20,
                'v1': -1.2,
                'v2': 18}
# auxiliary helper function(s) -- function name: ([func signature], definition)
DSargs.fnspecs  = {'minf': (['v'], '0.5 * (1 + tanh( (v-v1)/v2 ))') }
# rhs of the differential equation, including dummy variable w
DSargs.varspecs = {'v': '( i + gl * (vl - v) - gca * minf(v) * (v-vca) )/c',
                   'w': 'v-w' }
# initial conditions
DSargs.ics      = {'v': 0, 'w': 0 }
