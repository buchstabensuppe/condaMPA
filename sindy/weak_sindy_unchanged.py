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
from averaging_measurements_for_noise_reduction import average_noisy_measurement
from CSTR1 import simCSTR1
from MPIR_callable_function import MPI_reactor
import pysindy as ps

# generating data:
seconds = 1 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.001 #[s]
dt_time_seconds = int(seconds/dt)
filter = False

# Simulation auswählen: basic batch reactor = 1, MPI CSTR = 2, 3 = MPI CSTR but average of multiple measurements
reactor_choice = 2
if reactor_choice == 1:
    n_variables = 3
    x0s = [100, 50, 10]
    x0 = x0s
    data = simCSTR1(seconds, dt, n_variables, x0s)
    u_train = data

if reactor_choice == 2:
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.6, 0.15, 0, 0])
    data_raw = MPI_reactor(seconds, dt_time_seconds, x0, dont_plot = True)
    data_raw_test = MPI_reactor(seconds, dt_time_seconds, x0_test, dont_plot = True)
    x0s = data_raw[0]
    x0s_test = data_raw_test[0]
    data_tmp = np.array(data_raw)
    data_tmp_test = np.array(data_raw_test)
    data = np.zeros((dt_time_seconds, 4))
    data_test = np.zeros((dt_time_seconds,4))
    for i in range(dt_time_seconds):
        data[i] = [data_tmp[0, i], data_tmp[1, i], data_tmp[2, i], data_tmp[3, i]]
        data_test[i] = [data_tmp_test[0, i], data_tmp_test[1, i], data_tmp_test[2, i], data_tmp_test[3, i]]
    u_train = data
    u_test = data_test

if reactor_choice == 3:
    average_of_n = 100
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.6, 0.15, 0, 0])
    u_dot_clean, u_clean, u_train, u_test, u_dot = average_noisy_measurement(average_of_n, seconds, dt_time_seconds, x0)

t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])


# applying noise
if reactor_choice != 3:
    rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
    u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
    u_clean = u_test
    u_train = u_train + np.random.normal(0, rmse / 80.0, u_train.shape)  # Add 20% noise
    rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
    u_test = u_test + np.random.normal(0, rmse / 80.0, u_test.shape)  # Add 20% noise
    u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)

# setting negative values to zero: a *= (a>0)
u_train *= (u_train > 0)
u_test *= (u_test > 0)
u_dot *= (u_dot > 0)

# applying noise filters:
if filter == True:
    from scipy.signal import medfilt
    # Example usage
    u_train_filtered = medfilt(u_train, kernel_size=5)
    u_test_filtered = medfilt(u_test, kernel_size=5)
    u_train = u_train_filtered
    u_test = u_test_filtered

# Same library terms as before
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# Scan over the number of integration points and the number of subdomains
n = 10
errs = np.zeros((n))
K_scan = np.linspace(20, 2000, n, dtype=int) #ursprünglich: stop = 2000, höhere Werte liefern bessere Ergebnisse
for i, K in enumerate(K_scan):
    ode_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_grid=t_train,
        include_bias=True,
        is_uniform=True,
        K=2,
    )
    opt = ps.SR3(
        threshold=0.0001, #Standard war 0.05, appearently deutlich bessere Ergebnisse mit geringerem Threshold
        thresholder="l0",
        max_iter=1000,
        normalize_columns=True,
        tol=1e-1,
    )
    u_dot_train_integral = ode_lib.convert_u_dot_integral(u_train)
    u_valid_train_integral = ode_lib.convert_u_dot_integral(u_test)

    # Instantiate and fit the SINDy model with the integral of u_dot
    model = ps.SINDy(feature_library=ode_lib, optimizer=opt)
    model.fit(u_train, quiet=True)
    validate_with_test_data = False
    if validate_with_test_data == True:
        errs[i] = np.sqrt(
            (
                np.sum((u_valid_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
                / np.sum(u_valid_train_integral ** 2)
            )
            / u_valid_train_integral.shape[0]
        )
    else:
        errs[i] = np.sqrt(
            (
                np.sum((u_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
                / np.sum(u_dot_train_integral ** 2)
            )
            / u_dot_train_integral.shape[0]
        )
    model.print()

#ploting error graph
plt.title('Convergence of weak SINDy, hyperparameter scan', fontsize=12)
plt.plot(K_scan,errs)
plt.xlabel('Number of subdomains', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.show()

from symbolic_model_to_simulation_2 import simulate_sindy_result_2

simulate_sindy_result_2(model.coefficients(), x0, seconds, dt_time_seconds)

breakbreak = True