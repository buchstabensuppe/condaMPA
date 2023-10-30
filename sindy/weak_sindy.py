import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from pysindy.utils.odes import lorenz
import CSTR1
from CSTR1 import simCSTR1

import pysindy as ps

# Ignore matplotlib deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

# integration keywords for solve_ivp, typically needed for chaotic systems
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

###########################################
################### 2 #####################
###########################################

# Generate measurement data
# dt = 0.002
# t_train = np.arange(0, 10, dt)
# t_train_span = (t_train[0], t_train[-1])
# u0_train = [-8, 8, 27]
# u_train = solve_ivp(
#     lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
# ).y.T
#CSTR Simulation aufrufen und Simulationsdaten importeren

seconds =int(5) #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 10]
data = simCSTR1(seconds, dt, n_variables, x0s)

# mein CSTR in original namen übersetzen

t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])
u_train = data

# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
model = ps.SINDy()
model.fit(u_train, x_dot=u_dot, t=dt)
model.print()

# Define weak form ODE library
# defaults to derivative_order = 0 if not specified,
# and if spatial_grid is not specified, defaults to None,
# which allows weak form ODEs.
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
ode_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    spatiotemporal_grid=t_train,
    is_uniform=True,
    K=100,
)

# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.0001, thresholder="l1", max_iter=10000, normalize_columns=True, tol=1e-6
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()