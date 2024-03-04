# -*- coding: utf-8 -*-
"""
Created

@author: Giuseppe Armenise
example armax mimo
case 3 outputs x 4 inputs

"""
from __future__ import division

from past.utils import old_div

# Checking path to access other files
try:
    from sippy import *
except ImportError:
    import sys, os

    sys.path.append(os.pardir)
    from sippy import *

import numpy as np
import control.matlab as cnt
from sippy import functionset as fset



## generating data from MPI reactor simulation:
from sindy.Parameter_PDE_CH4 import data, update_dependent_values
from MPIR_callable_function import MPI_reactor
from sindy.CSTR1 import simCSTR1
c_in = data['x_in'] * data['p_R'] / (data['R'] * data['T_gas_in'])
seconds =1 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.01 #[s]
n_variables = 3
seconds = 1 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.0002 #[s]
dt_time_seconds = int(seconds/dt)

# Simulation auswählen: basic batch reactor = 1, MPI CSTR = 2
reactor_choice = 2
if reactor_choice == 1:
    n_variables = 3
    x0s = [100, 50, 10]
    data = simCSTR1(seconds, dt, n_variables, x0s)

if reactor_choice == 2:
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.75, 0.3, 0, 0])
    data_raw = MPI_reactor(seconds, c_in, dt_time_seconds, x0)
    data_raw_test = MPI_reactor(seconds, c_in, dt_time_seconds, x0_test)
    x0s = data_raw[0]
    x0s_test = data_raw_test[0]
    data = np.array(data_raw)
    data_test = np.array(data_raw_test)
    u = np.zeros((4, dt_time_seconds))
    for i in range(dt_time_seconds):
        u[0, i] = c_in[0]
        u[1, i] = c_in[1]
        u[2, i] = c_in[2]
        u[3, i] = c_in[3]
    x0s = data[0]
    x0s_test = data_test[0]


# 4*3 MIMO system
# generating transfer functions in z-operator.
var_list = [50., 100., 1.]
ts = 1.

NUM11 = [4, 3.3, 0., 0.]
NUM12 = [10, 0., 0.]
NUM13 = [7.0, 5.5, 2.2]
NUM14 = [-0.9, -0.11, 0., 0.]
DEN1 = [1., -0.3, -0.25, -0.021, 0., 0.]  #
H1 = [1., 0.85, 0.32, 0., 0., 0.]
na1 = 3
nb11 = 2
nb12 = 1
nb13 = 3
nb14 = 2
th11 = 1
th12 = 2
th13 = 2
th14 = 1
nc1 = 2
#
DEN2 = [1., -0.4, 0., 0., 0.]
NUM21 = [-85, -57.5, -27.7]
NUM22 = [71, 12.3]
NUM23 = [-0.1, 0., 0., 0.]
NUM24 = [0.994, 0., 0., 0.]
H2 = [1., 0.4, 0.05, 0., 0.]
na2 = 1
nb21 = 3
nb22 = 2
nb23 = 1
nb24 = 1
th21 = 1
th22 = 2
th23 = 0
th24 = 0
nc2 = 2
#
DEN3 = [1., -0.1, -0.3, 0., 0.]
NUM31 = [0.2, 0., 0., 0.]
NUM32 = [0.821, 0.432, 0.]
NUM33 = [0.1, 0., 0., 0.]
NUM34 = [0.891, 0.223]
H3 = [1., 0.7, 0.485, 0.22, 0.]
na3 = 2
nb31 = 1
nb32 = 2
nb33 = 1
nb34 = 2
th31 = 0
th32 = 1
th33 = 0
th34 = 2
nc3 = 3

# SISO transfer functions (G and H)
g_sample11 = cnt.tf(NUM11, DEN1, ts)
g_sample12 = cnt.tf(NUM12, DEN1, ts)
g_sample13 = cnt.tf(NUM13, DEN1, ts)
g_sample14 = cnt.tf(NUM14, DEN1, ts)

g_sample22 = cnt.tf(NUM22, DEN2, ts)
g_sample21 = cnt.tf(NUM21, DEN2, ts)
g_sample23 = cnt.tf(NUM23, DEN2, ts)
g_sample24 = cnt.tf(NUM24, DEN2, ts)

g_sample31 = cnt.tf(NUM31, DEN3, ts)
g_sample32 = cnt.tf(NUM32, DEN3, ts)
g_sample33 = cnt.tf(NUM33, DEN3, ts)
g_sample34 = cnt.tf(NUM34, DEN3, ts)

H_sample1 = cnt.tf(H1, DEN1, ts)
H_sample2 = cnt.tf(H2, DEN2, ts)
H_sample3 = cnt.tf(H3, DEN3, ts)

# time
tfin = 400
npts = int(old_div(tfin, ts)) + 1
Time = np.linspace(0, tfin, npts)

#INPUT#
Usim = np.zeros((4, npts))
Usim_noise = np.zeros((4, npts))
[Usim[0, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [-0.33, 0.1])
[Usim[1, :],_,_] = fset.GBN_seq(npts, 0.03)
[Usim[2, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [2.3, 5.7])
[Usim[3, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [8., 11.5])

# Adding noise
err_inputH = np.zeros((4, npts))
err_inputH = fset.white_noise_var(npts, var_list)

err_outputH1, Time, Xsim = cnt.lsim(H_sample1, err_inputH[0, :], Time)
err_outputH2, Time, Xsim = cnt.lsim(H_sample2, err_inputH[1, :], Time)
err_outputH3, Time, Xsim = cnt.lsim(H_sample3, err_inputH[2, :], Time)

# OUTPUTS
Yout = np.zeros((3, npts))
# Noise-free output
Yout11, Time, Xsim = cnt.lsim(g_sample11, Usim[0, :], Time)
Yout12, Time, Xsim = cnt.lsim(g_sample12, Usim[1, :], Time)
Yout13, Time, Xsim = cnt.lsim(g_sample13, Usim[2, :], Time)
Yout14, Time, Xsim = cnt.lsim(g_sample14, Usim[3, :], Time)
Yout21, Time, Xsim = cnt.lsim(g_sample21, Usim[0, :], Time)
Yout22, Time, Xsim = cnt.lsim(g_sample22, Usim[1, :], Time)
Yout23, Time, Xsim = cnt.lsim(g_sample23, Usim[2, :], Time)
Yout24, Time, Xsim = cnt.lsim(g_sample24, Usim[3, :], Time)
Yout31, Time, Xsim = cnt.lsim(g_sample31, Usim[0, :], Time)
Yout32, Time, Xsim = cnt.lsim(g_sample32, Usim[1, :], Time)
Yout33, Time, Xsim = cnt.lsim(g_sample33, Usim[2, :], Time)
Yout34, Time, Xsim = cnt.lsim(g_sample34, Usim[3, :], Time)

# Total output
Ytot1 = Yout11 + Yout12 + Yout13 + Yout14
Ytot2 = Yout21 + Yout22 + Yout23 + Yout24
Ytot3 = Yout31 + Yout32 + Yout33 + Yout34

Ytot = np.zeros((3, npts))

Ytot[0, :] = (Ytot1 + err_outputH1).squeeze()
Ytot[1, :] = (Ytot2 + err_outputH2).squeeze()
Ytot[2, :] = (Ytot3 + err_outputH3).squeeze()

##identification parameters
if reactor_choice == 0:
    ordersna = [na1, na2, na3]
    ordersnb = [[nb11, nb12, nb13, nb14], [nb21, nb22, nb23, nb24], [nb31, nb32, nb33, nb34]]
    ordersnc = [nc1, nc2, nc3]
    theta_list = [[th11, th12, th13, th14], [th21, th22, th23, th24], [th31, th32, th33, th34]]
if reactor_choice == 2:
    ordersna = [1, 1, 1, 1]
    ordersnb = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ordersnc = [1, 1, 1, 1]
    theta_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]]

# IDENTIFICATION STAGE
# TESTING ARMAX models
# iterative LLS
num_iterations = 2000000
# Id_ARMAXi = system_identification(data, u, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list],
#                                   max_iterations=num_iterations, centering = 'MeanVal')  #
# optimization-based
Id_ARMAXo = system_identification(data, u, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list],
                                  ARMAX_mod = 'OPT', max_iterations=num_iterations, centering = 'None')  #
# recursive LLS
Id_ARMAXr = system_identification(data, u, 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list],
                                  ARMAX_mod = 'RLLS', max_iterations=num_iterations, centering = 'InitVal')  #

# output of the identified model
# Yout_ARMAXo = Id_ARMAXo.Yid
Yout_ARMAXr = Id_ARMAXr.Yid
# print('Id_ARMAXi', Id_ARMAXi.G)
print('Id_ARMAXo', Id_ARMAXo.G)
print('Id_ARMAXr', Id_ARMAXr.G)
Yv_armaxr = fset.validation(Id_ARMAXr,u,Ytot_v,Time, centering = 'InitVal')


# print('YoutARMAXr', Id_ARMAXr)
# print('Yout_ARMAXo', Yout_ARMAXo)
# print('Yout_ARMAXo', Yout_ARMAXi)
######plots
#
import matplotlib.pyplot as plt

# U
plt.close('all')
# plt.figure(0)
# plt.subplot(4, 1, 1)
# plt.plot(Time, Usim[0, :])
# plt.grid()
# plt.ylabel("Input 1 - GBN")
# plt.xlabel("Time")
# plt.title("Input (Switch probability=0.03) (identification data)")
#
# plt.subplot(4, 1, 2)
# plt.plot(Time, Usim[1, :])
# plt.grid()
# plt.ylabel("Input 2 - GBN")
# plt.xlabel("Time")
#
# plt.subplot(4, 1, 3)
# plt.plot(Time, Usim[2, :])
# plt.ylabel("Input 3 - GBN")
# plt.xlabel("Time")
# plt.grid()
#
# plt.subplot(4, 1, 4)
# plt.plot(Time, Usim[3, :])
# plt.ylabel("Input 4 - GBN")
# plt.xlabel("Time")
# plt.grid()
#
# # Y
# plt.figure(1)
# plt.subplot(3, 1, 1)
# plt.plot(Time, Ytot[0, :])
# plt.plot(Time, Yout_ARMAXi[0,:])
# plt.plot(Time, Yout_ARMAXo[0,:])
# plt.plot(Time, Yout_ARMAXr[0,:])
# plt.ylabel("y$_1$,out")
# plt.grid()
# plt.xlabel("Time")
# plt.title("Identification data")
# plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
#
# plt.subplot(3, 1, 2)
# plt.plot(Time, Ytot[1, :])
# plt.plot(Time, Yout_ARMAXi[1,:])
# plt.plot(Time, Yout_ARMAXo[1,:])
# plt.plot(Time, Yout_ARMAXr[1,:])
# plt.ylabel("y$_2$,out")
# plt.grid()
# plt.xlabel("Time")
# plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
#
# plt.subplot(3, 1, 3)
# plt.plot(Time, Ytot[2, :])
# plt.plot(Time, Yout_ARMAXi[2,:])
# plt.plot(Time, Yout_ARMAXo[2,:])
# plt.plot(Time, Yout_ARMAXr[2,:])
# plt.ylabel("y$_3$,out")
# plt.grid()
# plt.xlabel("Time")
# plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
#

### VALIDATION STAGE

# time
tfin = 1
npts = int(old_div(tfin, ts)) + 1
Time = np.linspace(0, tfin, npts)

# (NEW) INPUTS
U_valid = u
# U_valid = np.zeros((4, npts))
# Usim_noise = np.zeros((4, npts))
# [U_valid[0, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [0.33, 0.7])
# [U_valid[1, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [-2., -1.])
# [U_valid[2, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [1.3, 2.7])
# [U_valid[3, :],_,_] = fset.GBN_seq(npts, 0.03, Range = [1., 5.2])
# Noise
err_inputH = np.zeros((4, npts))
err_inputH = fset.white_noise_var(npts, var_list)
err_outputH1, Time, Xsim = cnt.lsim(H_sample1, err_inputH[0, :], Time)
err_outputH2, Time, Xsim = cnt.lsim(H_sample2, err_inputH[1, :], Time)
err_outputH3, Time, Xsim = cnt.lsim(H_sample3, err_inputH[2, :], Time)
# Total Output
Yout = np.zeros((3, npts))
Yout11, Time, Xsim = cnt.lsim(g_sample11, U_valid[0, :], Time)
Yout12, Time, Xsim = cnt.lsim(g_sample12, U_valid[1, :], Time)
Yout13, Time, Xsim = cnt.lsim(g_sample13, U_valid[2, :], Time)
Yout14, Time, Xsim = cnt.lsim(g_sample14, U_valid[3, :], Time)
Yout21, Time, Xsim = cnt.lsim(g_sample21, U_valid[0, :], Time)
Yout22, Time, Xsim = cnt.lsim(g_sample22, U_valid[1, :], Time)
Yout23, Time, Xsim = cnt.lsim(g_sample23, U_valid[2, :], Time)
Yout24, Time, Xsim = cnt.lsim(g_sample24, U_valid[3, :], Time)
Yout31, Time, Xsim = cnt.lsim(g_sample31, U_valid[0, :], Time)
Yout32, Time, Xsim = cnt.lsim(g_sample32, U_valid[1, :], Time)
Yout33, Time, Xsim = cnt.lsim(g_sample33, U_valid[2, :], Time)
Yout34, Time, Xsim = cnt.lsim(g_sample34, U_valid[3, :], Time)
#
Ytot1 = Yout11 + Yout12 + Yout13 + Yout14
Ytot2 = Yout21 + Yout22 + Yout23 + Yout24
Ytot3 = Yout31 + Yout32 + Yout33 + Yout34
#
Ytot_v = np.zeros((3, npts))
#
Ytot_v[0, :] = (Ytot1 + err_outputH1).squeeze()
Ytot_v[1, :] = (Ytot2 + err_outputH2).squeeze()
Ytot_v[2, :] = (Ytot3 + err_outputH3).squeeze()

# ## Compute time responses for identified systems with new inputs

# ARMAX - ILLS
Yv_armaxi = fset.validation(Id_ARMAXi,U_valid,Ytot_v,Time, centering = 'MeanVal')

# ARMAX - OPT
Yv_armaxo = fset.validation(Id_ARMAXo,U_valid,Ytot_v,Time)

# ARMAX - RLLS
Yv_armaxr = fset.validation(Id_ARMAXr,U_valid,Ytot_v,Time, centering = 'InitVal')

# U
plt.figure(3)
plt.subplot(4, 1, 1)
plt.plot(Time, U_valid[0, :])
plt.grid()
plt.ylabel("Input 1 - GBN")
plt.xlabel("Time")
plt.title("Input (Switch probability=0.03) (validation data)")

plt.subplot(4, 1, 2)
plt.plot(Time, U_valid[1, :])
plt.ylabel("Input 2 - GBN")
plt.xlabel("Time")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(Time, U_valid[2, :])
plt.ylabel("Input 3 - GBN")
plt.xlabel("Time")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(Time, U_valid[3, :])
plt.ylabel("Input 4 - GBN")
plt.xlabel("Time")
plt.grid()

# Y
plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(Time, Ytot_v[0, :])
plt.plot(Time, Yv_armaxi[0,:])
plt.plot(Time, Yv_armaxo[0,:])
plt.plot(Time, Yv_armaxr[0,:])
plt.ylabel("y_1,out")
plt.grid()
plt.xlabel("Time")
plt.title("Validation data")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

plt.subplot(3, 1, 2)
plt.plot(Time, Ytot_v[1, :])
plt.plot(Time, Yv_armaxi[1,:])
plt.plot(Time, Yv_armaxo[1,:])
plt.plot(Time, Yv_armaxr[1,:])
plt.ylabel("y_2,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

plt.subplot(3, 1, 3)
plt.plot(Time, Ytot_v[2, :])
plt.plot(Time, Yv_armaxi[2,:])
plt.plot(Time, Yv_armaxo[2,:])
plt.plot(Time, Yv_armaxr[2,:])
plt.ylabel("y_3,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

plt.show()

print('Ytot_v', Ytot_v)

breakbreak = True