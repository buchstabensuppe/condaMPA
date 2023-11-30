#source: https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html#Test-weak-form-ODE-functionality-on-Lorenz-equation

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

seconds =5 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
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


t_span = np.linspace(0, 5, 100)

# das errechnete Modell plotten:
u_weak = model.predict(x0s, t_span)
plt.plot(t_train, u_weak[:,1], "b", label=r"$q$ weak form prediction")

# Plot läuft nicht, Rest schon, @Alex kannst du den Plot reparieren?

##### #3 on website, used here in the attempt to learn how to extract the ODEs out of the model
# Fazit: Hilft mir gar nicht mit der Frage, läuft aber butterweich. einzig unklar, was ich jetzt
# damit anfange, @Alex fragen ob auch ganz andere Differentialgleichungen okay sind, solange sie funktionieren

# Generate measurement data
# dt = 0.001
# t_train = np.arange(0, 10, dt)
# t_train_span = (t_train[0], t_train[-1])
# u0_train = [-8, 8, 27]
u0_test = [100, 50, 0]
data = simCSTR1(seconds, dt, n_variables, u0_test)
t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])
# u_test = #20 rand percent of data
# u_train = #other 80 persent of data


# u_train = solve_ivp(
#     lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
# ).y.T
#u_test = solve_ivp(
#    lorenz, t_train_span, u0_test, t_eval=t_train, **integrator_keywords).y.T

rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
u_clean = u_test
u_train = u_train + np.random.normal(0, rmse / 5.0, u_train.shape)  # Add 20% noise
rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
u_test = u_test + np.random.normal(0, rmse / 5.0, u_test.shape)  # Add 20% noise
u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)

# Same library terms as before
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# Scan over the number of integration points and the number of subdomains
n = 20
errs = np.zeros((n))
K_scan = np.linspace(20, 10000, n, dtype=int)
for i, K in enumerate(K_scan):
    ode_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t_train,
        include_bias=True,
        is_uniform=True,
        K=K,
    )
    #opt = optimizer
    opt = ps.SR3(
        threshold=0.1,
        thresholder="l0",
        max_iter=1000000,
        normalize_columns=True,
        tol=1e-9,
    )
    u_dot_train_integral = ode_lib.convert_u_dot_integral(u_train)

    # Instantiate and fit the SINDy model with the integral of u_dot
    model = ps.SINDy(feature_library=ode_lib, optimizer=opt)
    model.fit(u_train)
    errs[i] = np.sqrt(
        (
            np.sum((u_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
            / np.sum(u_dot_train_integral**2)
        )
        / u_dot_train_integral.shape[0]
    )
    print('final model', i, 'shown in plot:')
    model.print()

# printing model and plot to show convergence



plt.title("Convergence of weak SINDy, hyperparameter scan", fontsize=12)
plt.plot(K_scan, errs)
plt.xlabel("Number of subdomains", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show()