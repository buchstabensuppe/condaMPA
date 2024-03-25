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

def average_noisy_measurement(n, seconds, dt_time_seconds, x0):
    u_dot_clean_log = np.zeros((1000,4))
    u_clean_log = np.zeros((1000,4))
    u_train_log = np.zeros((1000,4))
    u_test_log = np.zeros((1000,4))
    u_dot_log = np.zeros((1000,4))
    for ititi in range(n):
        # generating data:
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

        t_train = np.arange(0, seconds, dt)
        t_train_span = (t_train[0], t_train[-1])
        u_train = data
        u_test = data_test

        # applying noise
        rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
        u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
        u_clean = u_test
        u_train = u_train + np.random.normal(0, rmse / 80.0, u_train.shape)  # Add 20% noise
        rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
        u_test = u_test + np.random.normal(0, rmse / 80.0, u_test.shape)  # Add 20% noise
        u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)

        u_dot_clean_log += u_dot_clean
        u_clean_log += u_clean
        u_train_log += u_train
        u_test_log += u_test
        u_dot_log += u_dot_clean

    return u_dot_clean_log/n, u_clean_log/n, u_train_log/n, u_test_log/n, u_dot_log/n
