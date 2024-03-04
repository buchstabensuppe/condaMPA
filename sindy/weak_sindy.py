#source: https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html#Test-weak-form-ODE-functionality-on-Lorenz-equation
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from pysindy.utils.odes import lorenz
import CSTR1
from CSTR1 import simCSTR1
from MPIR_callable_function import MPI_reactor
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


seconds = 1 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.001 #[s]
dt_time_seconds = int(seconds/dt)

# Simulation auswählen: basic batch reactor = 1, MPI CSTR = 2
reactor_choice = 2
if reactor_choice == 1:
    n_variables = 3
    x0s = [100, 50, 10]
    x0 = x0s
    data = simCSTR1(seconds, dt, n_variables, x0s)

if reactor_choice == 2:
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.6, 0.15, 0, 0])
    data_raw = MPI_reactor(seconds, dt_time_seconds, x0, dont_plot = False)
    data_raw_test = MPI_reactor(seconds, dt_time_seconds, x0_test, dont_plot = False)
    x0s = data_raw[0]
    x0s_test = data_raw_test[0]
    data_tmp = np.array(data_raw)
    data_tmp_test = np.array(data_raw_test)
    data = np.zeros((dt_time_seconds, 4))
    data_test = np.zeros((dt_time_seconds,4))
    for i in range(dt_time_seconds):
        data[i] = [data_tmp[0, i], data_tmp[1, i], data_tmp[2, i], data_tmp[3, i]]
        data_test[i] = [data_tmp_test[0, i], data_tmp_test[1, i], data_tmp_test[2, i], data_tmp_test[3, i]]

# mein CSTR in original namen übersetzen
# x0 = x0 * (1/35)
# data = (1/35) * data
t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])
u_train = data
u_test = data_test

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
    K=20,
)

# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.001, thresholder="l1", max_iter=10000, normalize_columns=True, tol=1e-4
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()

import symbolic_model_to_simulation
from symbolic_model_to_simulation import simulate_sindy_result

simulate_sindy_result(model.coefficients(), x0, seconds, dt_time_seconds)

# Instantiate and fit a non-weak SINDy model

############ oritinal model fitten, nötig für plot des weak models
# ode_lib = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
#     include_bias=True,
# )
# optimizer = ps.SR3(
#     threshold=100, thresholder="l0", max_iter=10000, normalize_columns=True, tol=1e-10
# )
# original_model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
# original_model.fit(u_train, t=dt, quiet=True)
# print("Regular model: ")
# original_model.print()
# err_not_weak = np.sqrt(
#     (np.sum((u_dot - optimizer.Theta_ @ optimizer.coef_.T) ** 2) / np.sum(u_dot**2))
#     / u_dot.shape[0]
# )
#
# u_pred = original_model.simulate(x0s, t=t_train, integrator="odeint")
#u_dot_pred = original_model.predict(u_test)
######## ab hier weak_sindy


# Plot sollte funktionieren äquivalent zu Block 5 aus: https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html
# das errechnete Modell simulieren

#u_weak_1 = model.predict(x0s)
# plt.plot(t_train, u_weak[:,1], "b", label=r"$q$ weak form prediction")

# Plot läuft nicht, Rest schon, @Alex kannst du den Plot reparieren?

##### #3 on website, used here in the attempt to learn how to extract the ODEs out of the model
# Fazit: Hilft mir gar nicht mit der Frage, läuft aber butterweich. einzig unklar, was ich jetzt
# damit anfange, @Alex fragen ob auch ganz andere Differentialgleichungen okay sind, solange sie funktionieren

# Generate measurement data
# dt = 0.001
# t_train = np.arange(0, 10, dt)
# t_train_span = (t_train[0], t_train[-1])
# u0_train = [-8, 8, 27]
#u0_test = [100, 50, 0]
#data = simCSTR1(seconds, dt, n_variables, u0_test)
#t_train = np.arange(0, seconds, dt)
#t_train_span = (t_train[0], t_train[-1])
# u_test = #20 rand percent of data
# u_train = #other 80 persent of data

##### splitting simulated data randomly for test and training (20/80)
##### not used, as weak sindy expects test and training data of same size
# from sklearn.model_selection import train_test_split
# x_train, x_test = train_test_split(data, test_size=0.5)
# print(x_train, x_test)
#
# u_test = x_test
# u_train = x_train
u_train = data
u_test = data_test
# u_train = solve_ivp(
#     lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
# ).y.T
#u_test = solve_ivp(
#    lorenz, t_train_span, u0_test, t_eval=t_train, **integrator_keywords).y.T

u_test = u_train
#
# # noise implementation weak_sindy docu:
# rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
# u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
# u_clean = u_test
# u_train = u_train + np.random.normal(0, rmse / 300.0, u_train.shape)  # Add 20% noise
# print('u_train with noise:', u_train)
# rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
# u_test = u_test + np.random.normal(0, rmse / 300.0, u_test.shape)  # Add 20% noise
# u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)
#
# noise implementation alex
print(u_train)
for j in range(n_variables):
    for i in range(len(u_train)):
        noise = np.random.normal(0, 0.001, u_train[i,j].shape)
        u_train[i,j] = u_train[i,j] * (1 + noise)
print(u_train)
u_valid = u_train
print(u_valid)
for j in range(n_variables):
    for i in range(len(u_train)):
        noise = np.random.normal(0, 0.001, u_train[i,j].shape)
        u_valid[i,j] = u_train[i,j] * (1 + noise)
print(u_valid)

# run weak sindy again, with the noise defined above this line, but without hyperparameter tuning below:
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]
ode_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    spatiotemporal_grid=t_train,
    is_uniform=True,
    K=20,
)

# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.001, thresholder="l1", max_iter=10000, normalize_columns=True, tol=1e-2
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()

import symbolic_model_to_simulation
from symbolic_model_to_simulation import simulate_sindy_result

simulate_sindy_result(model.coefficients(), x0, seconds, dt_time_seconds)

######### Hyperparameter tuing part below ####################

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
        K=20, # soll ich laut google bard varrieren, um overfitting zu reduzieren. Laut bard mit 2 oder 3 starten und dann langsam erhöhen. Ursprünglich: K=K
    )
    #opt = optimizer
    opt = ps.SR3(
        threshold=0.3,
        thresholder="l1",
        max_iter=100000,
        normalize_columns=True,
        tol=1e-1, #lower tol -> more overfitting, higher tol -> faster convergance but higher error
    )
    u_dot_train_integral = ode_lib.convert_u_dot_integral(u_train)

    # Instantiate and fit the SINDy model with the integral of u_dot
    model = ps.SINDy(feature_library=ode_lib, optimizer=opt)
    model.fit(u_train)
    u_valid_train_integral = ode_lib.convert_u_dot_integral(u_valid)
    errs[i] = np.sqrt(

        (
            np.sum((u_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
            #/ np.sum(u_dot_train_integral**2)
        ))/ u_dot_train_integral.shape[0]

    print('final model', i, 'shown in plot:')
    model.print()
    breakbreak = True
# printing model and plot to show convergence



plt.title("Convergence of weak SINDy, hyperparameter scan", fontsize=12)
plt.plot(K_scan, errs)
plt.xlabel("Number of subdomains", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show()

breakbreak = True

# Potential parameters to vary for weak_sindy:
# K: 'K determines the complexity of the model by specifying the number of basis functions used to represent the underlying dynamics.'
# Thresholder: L1 best to avoid overfitting according to google bard
# threshold
# error

# Hyperparamter tuning: Not possible without working error metrics

# -> first goal: finding a working error metric