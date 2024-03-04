import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from scipy import integrate
from beautifultable import BeautifulTable
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import least_squares
import time
import plotly.express as px
import plotly.graph_objects as go
import reaction_rate_alex
from reaction_rate_alex import reaction_rate
import Parameter_PDE_CH4
from Parameter_PDE_CH4 import data, update_dependent_values

#### testing reaction rate r####

#r = reaction_rate(330, [0,0,0,0], 2E5)
#print(r)


# Dictionary of Parameters
p = {

# Inlet Concentrations
'cain': 10,
'cbin': 2.5,
'ccin': 0,
'cdin': 0,

# Frequency Factors
'k01': 5.4e10,
'k02': 4.6e17,
'k03': 5.0e7,

# Activation Energies
'ea1': 7.5e4,
'ea2': 1.2e5,
'ea3': 5.5e4,

# Exponents
'n1': 1.1,
'n2': 1,
'n3': 1,

# Gasconstant, Temperature, Cross Section, Length
'r': 8.3145,
'temp': 330,
'a': 0.1,
'q': 0.1,
'l': 50,
'p_R': 2E5,
'R': 8.3145,
'T_gas_in': 270+273.15,
# added
# 'nu_a':
# 'nu_b':
# 'nu_c':
# 'nu_d':
# 'rho':

#seconds of simulation:
'seconds': 5,
'L_r': 2,
"D_r": 0.01,  # Inner diameter reactor in m


}

# Vector of Initial Concentrations
p['cin']  =  data['x_in'] * data['p_R'] / (data['R'] * data['T_gas_in'])


############### Temperatur in bereitgestellten Daten konstant?

# Auxiliary function for Arrhenius-Calculation
# def arrhenius(k0, ea, p):
#     k = k0 * np.exp(- (ea / (p['r'] * data['t_gas_in'])))
#     return k



# ODE-System
def ode_system(t, c, p):

    update_dependent_values()
    # Unpacking the Concentrations
    ca = np.array(c[0])
    cb = np.array(c[1])
    cc = np.array(c[2])
    cd = np.array(c[3])

    x =np.array([0.0, 0.0, 0.0 ,0.0])
    ######## @Alex not so sure if this is fine, but need x somehow

    sum_c = c[0] + c[1] + c[2] + c[3]

    for i in range(4):
        if c[i] <= 0:
            c[i] = 0

    x[0] = ca/sum_c

    x[1] = cb/sum_c

    x[2] = cc/sum_c

    x[3] = cd/sum_c

    # Calculation of the rate constant
    #k1 = arrhenius(p['k01'], p['ea1'], p)
    #k2 = arrhenius(p['k02'], p['ea2'], p)
    #k3 = arrhenius(p['k03'], p['ea3'], p)

    # Calculation of the rates


    #r1 = k1 * np.power(ca, p['n1'])
    #r2 = k2 * np.power(cb, p['n2'])
    #r3 = k3 * np.power(cb, p['n3'])

    # Calculation of the gradients

    #dc_i / dt = (1 / tau)(c_i_in - ci) + nu_i * rho * r
    #tau = tau = Volumen/Volumenstrom = Querschnitt * Länge/ Volumenstrom

    ########### @Alex, breauchst du für das Volumen nicht pi? dafuq

    #tau = data['V_r']*data['F_in']


    # n_in = data['n_in']
    # c_in = np.array([0, 0, 0, 0])
    # for i in range(4):
    #     c_in[i] = n_in[i]*V_r

    #c_in = n_in_total * x_in

    #dc_i / dt = (1 / tau)(c_i_in - ci) + nu_i * rho * r
    # dcadt = (1 / tau)(c_in[0] - ca) + p['nu_a'] * p['rho'] * r
    # dcbdt = (1 / tau)(c_in[1] - cb) + p['nu_b'] * p['rho'] * r
    # dccdt = (1 / tau)(c_in[2] - cc) + p['nu_c'] * p['rho'] * r
    # dcddt = (1 / tau)(c_in[3] - cd) + p['nu_d'] * p['rho'] * r


    #tau = 0.2
    # dcadz = (1 / tau)*(c_in[0] - ca) + ny[0] * 1032 * r
    # dcbdz = (1 / tau)*(c_in[1] - cb) + ny[1] * 1032 * r
    # dccdz = (1 / tau)*(c_in[2] - cc) + ny[2] * 1032 * r #data['rho_cat'] * r
    # dcddz = (1 / tau)*(c_in[3] - cd) + ny[3] * 1032 * r

    x0 = c[0]
    x1 = c[1]
    x2 = c[2]
    x3 = c[3]


    # weak sindy result for
    # dcadz = 1240.969 * 1 + -144.725 * x0 + -194.589 * x1 + 334.655 * x2 + 173.464 * x3 + 3.156 * x0 * x0 + -2.821 * x1 * x1 + -7.745 * x2 * x2 + 3.312 * x3 * x3 + 6.342 * x0 * x1 + -8.013 * x0 * x2 + -6.082 * x0 * x3 + -2.073 * x1 * x2 + -0.940 * x1 * x3 + -0.245 * x2 * x3
    # dcbdz = 2403.040 * 1 + -137.797 * x0 + 6.226 * x1 + -81.575 * x2 + 43.940 * x3 + 1.870 * x0 * x0 + 4.226 * x1 * x1 + -0.221 * x2 * x2 + -1.371 * x3 * x3 + -0.342 * x0 * x1 + 3.035 * x0 * x2 + -0.512 * x0 * x3 + 0.807 * x1 * x2 + 0.479 * x1 * x3 + -3.253 * x2 * x3
    # dccdz = 1601.529 * 1 + -89.487 * x0 + 129.126 * x1 + 32.148 * x2 + -91.630 * x3 + 1.153 * x0 * x0 + 2.232 * x1 * x1 + -1.854 * x2 * x2 + 1.244 * x3 * x3 + -3.309 * x0 * x1 + 0.078 * x0 * x2 + 3.100 * x0 * x3 + -4.223 * x1 * x2 + -0.165 * x1 * x3 + -4.795 * x2 * x3
    # dcddz = -3495.433 * 1 + 209.507 * x0 + 72.447 * x1 + -1.508 * x2 + 5.710 * x3 + -3.087 * x0 * x0 + -2.497 * x1 * x1 + 0.263 * x2 * x2 + 2.601 * x3 * x3 + -1.108 * x0 * x1 + -0.126 * x0 * x2 + -0.821 * x0 * x3 + -2.872 * x1 * x2 + -5.129 * x1 * x3 + 0.362 * x2 * x3

    # dcadz = 5280.549 * 1 + -210.992 * x0 + 205.560 * x1 + -0.624 * x2 + 2.091 *x0*x0 + 2.426 *x1*x1 + 0.001 *x2*x2 + -4.304 *x0*x1 + 0.039* x0*x2 + 0.038* x1*x2
    # dcbdz = -2481.493 *1 + 84.814 *x0 + -71.131 *x1 + 12.729 *x2 + -0.699 *x0*x0 + -0.516 *x1*x1 + 0.001 *x2*x2 + 1.163 *x0*x1 + -0.262 *x0*x2 + 0.246 *x1*x2
    # dccdz = -3566.194 *1 + 128.742* x0 + -111.427* x1 + 11.783 *x2 + -1.138 *x0*x0 + -1.019 *x1*x1 + -0.003 *x2*x2 + 2.059* x0*x1 + -0.254 *x0*x2 + 0.215 *x1*x2
    # dcddz = 0
    # weak sindy result for
    # dcadz = -24.622 * x1 * x1
    # dcbdz = 0.196 * x2 + 0.196 * x3 + 39.675 * x1 * x1 + 23.215 * x2 * x2 + 23.215 * x3 * x3 + -4.647 * x0 * x2 + -4.647 * x0 * x3 + 23.215 * x2 * x3
    # dccdz = 49.243 * x1 * x1
    # dcddz = 49.243 * x1 * x1

    dcadz = 6273.718 * 1 + -648.450 * x0 + -757.657 * x1 + 1327.217 * x2 + 687.269 * x3 + 13.505 * x0 * x0 + -12.769 * x1+x1 + -30.303 *x2*x2 + 13.541 *x3*x3 + 23.995 *x0*x1 + -31.619 *x0*x2 + -24.010 *x0*x3 + -7.521 *x1*x2 + -0.765 *x1*x3 + -2.262 *x2*x3
    dcbdz = 11954.259 * 1 + -689.246 * x0 + 91.225 * x1 + -375.562 * x2 + 161.959 * x3 + 9.538 * x0* x0 + 13.764 *x1*x1 + 0.559 *x2*x2 + -5.017 *x3*x3 + -4.612 *x0*x1 + 13.387 *x0*x2 + -1.372 *x0*x3 + 5.123 *x1*x2 + 8.933 *x1*x3 + -16.197 *x2*x3
    dccdz = 5026.515 * 1 + -277.012 * x0 + 477.352 * x1 + 154.129 * x2 + -354.809 * x3 + 3.406 * x0 * x0 + 11.169* x1*x1 + -8.448 *x2*x2 + 4.413 *x3*x3 + -11.355 *x0*x1 + -0.125 *x0*x2 + 11.882 *x0*x3 + -17.904 *x1*x2 + -4.754 *x1*x3 + -17.557 *x2*x3
    dcddz = -14960.498 * 1 + 895.601 * x0 + 261.676 * x1 + 13.606 * x2 + 21.526 * x3 + -13.176 * x0 * x0 + -8.606 *x1*x1 + 0.408 *x2*x2 + 10.123 *x3*x3 + -3.127 *x0*x1 + -0.917 *x0*x2 + -3.604 *x0*x3 + -12.228 *x1*x2 + -23.336 *x1*x3 + 2.648 *x2*x3
    # #dcadz = -(p['a'] / p['q']) * r1
    # dcbdz = (p['a'] / p['q']) * (r1 - r2 - r3)
    # dccdz = (p['a'] / p['q']) * r2
    # dcddz = (p['a'] / p['q']) * r3
    # dcadz = -0.033 * x1 + 3.157 * x1* x1 + -0.304 *x0*x1 + -1.724 *x1*x2 + -1.724 *x1*x3
    # dcbdz = -0.132 * x1 + 12.627 * x1*x1 + -1.216 *x0*x1 + -6.897 *x1*x2 + -6.897 *x1*x3
    # dccdz = 0.066 * x1 + -6.314 *x1*x1 + 0.608 *x0*x1 + 3.448 *x1*x2 + 3.448 *x1*x3
    # dcddz = 0.066 * x1 + -6.314 *x1*x1 + 0.608 *x0*x1 + 3.448 *x1*x2 + 3.448 *x1*x3

    # mpi reactor with settings from weak sindy (model 0 shown in plot)
    # dcadz = -11.076 *x1*x1
    # dcbdz = 0.097 *x1 + -28.654 *x1*x1 + -3.934 *x0*x1 + 14.758 *x1*x2 + 14.758 *x1*x3
    # dccdz = 22.152 *x1*x1
    # dcddz = 22.152 *x1*x1

    # mpi reactor with results from normal sindy (main_sindy) (sieht gut aus)
    # dcadz = -0.018 *x1 + -0.112 *x0*x1 + -1.018 *x1*x2 + -1.018 *x1*x3
    # dcbdz = -0.054 *x1 + 2.668 *x1*x1 + -1.117 *x0*x1 + -1.572 *x1*x2 + -1.572 *x1*x3
    # dccdz = 0.027 *x1 + -1.334 *x1*x1 + 0.559 *x0*x1 + 0.786 *x1*x2 + 0.786 *x1*x3
    # dcddz = 0.027 *x1 + -1.334 *x1*x1 + 0.559 *x0*x1 + 0.786 *x1*x2 + 0.786 *x1*x3

    # weak sindy mit hyperparameter tuning, threshold = 0.5, l1
    # dcadz = -8423.395 *1 + 499.116 *x0 + -7.372 *x0*x0 + -1.324 *x1*x3
    # dcbdz = -16.998 *x1*x3 -22 *x1*x3
    # dccdz = 8.123 *x1*x3
    # dcddz = 8.428 *x1*x3

    #weak sindy output vor hyperparameter tuning
    # dcadz = -0.405 * x1 + 2.188 *x1*x1 + 0.001 *x2*x2 + 0.001 *x3*x3 + -2.886 *x1*x2 + -2.886 *x1*x3 + 0.001 *x2*x3
    # dcbdz = -0.136 * x1 + 13.327 *x1*x1 + 0.005 *x2*x2 + 0.005 *x3*x3 + -1.186 *x0*x1 + -7.266 *x1*x2 + -7.266 *x1*x3 + 0.005 *x2*x3
    # dccdz  = 0.068 * x1 + -6.663 *x1*x1 + -0.002 *x2*x2 + -0.002 *x3*x3 + 0.593 *x0*x1 + 3.633 *x1*x2 + 3.633 *x1*x3 + -0.002 *x2*x3
    # dcddz = 0.068 * x1 + -6.663 *x1*x1 + -0.002 *x2*x2 + -0.002 *x3*x3 + 0.593 *x0*x1 + 3.633 *x1*x2 + 3.633 *x1*x3 + -0.002 *x2*x3

    # dcadz = -18039.553 *1 + -813.502 *x0 + -1.395 *x1 + 9238.175 *x2 + 4925.538 *x3 + 42.305 *x0*x0 + -56.443 *x1*x1 + 84.664 *x2*x2 + 63.147 *x3*x3 + -5.822 *x0*x1 + -292.681 *x0*x2 + -156.679 *x0*x3 + 43.026 *x1*x3 + -58.061 *x2*x3
    # dcbdz = -407873.987 *1 + 19458.262 *x0 + -14494.879 *x1 + 28626.575 *x2 + 9161.035 *x3 + -217.727 *x0*x0 + 114.594 *x1*x1 + 311.365 *x2*x2 + -192.208 *x3*x3 + 354.671 *x0*x1 + -919.640 *x0*x2 + -198.425 *x0*x3 + 231.542 *x1*x2 + 373.427 *x1*x3 + -175.850 *x2*x3
    # dccdz = 212983.367 *1 + -8427.590 *x0 + -3746.627 *x1 + -21646.269 *x2 + -10970.496 *x3 + 59.442 *x0*x0 + -26.926 *x1*x1 + -128.767 *x2*x2 + 73.656 *x3*x3 + 147.298 *x0*x1 + 690.153 *x0*x2 + 313.384 *x0*x3 + -43.599 *x1*x2 + -215.094 *x1*x3 + -37.132 *x2*x3
    # dcddz = -222188.436 *1 + 15745.143 *x0 + 10196.267 *x1 + -16970.388 *x2 + -849.961 *x3 + -275.376 *x0*x0 + -101.304 *x1*x1 + -218.627 *x2*x2 + -191.489 *x3*x3 + -230.279 *x0*x1 + 536.602 *x0*x2 + 42.973 *x0*x3 + -140.698 *x1*x2 + -432.096 *x1*x3 + 241.202 *x2*x3

    # Merging the Concentrations to one array
    dcdz = np.array([dcadz, dcbdz, dccdz, dcddz])

    return dcdz

# Solve the ODE


# Integration area
zspan = (0, 1)


# Define lambda-function
func = lambda t,c : ode_system(t,c,p)


# Solving the ODE-System for Initial Conditions
#sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45')#,t_eval=np.linspace(0, data['L_R'], 10))
sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45',t_eval=np.linspace(0, 1, 1000))

#zspan = (0, 50)
#sol = integrate.solve_ivp(func, zspan, [100, 50, 10, 0], method='RK45',t_eval=np.linspace(0, 50, 1000))

# Unpacking the Trajectories of the Concentrations
ca = sol.y[0, :]
cb = sol.y[1, :]
cc = sol.y[2, :]
cd = sol.y[3, :]

# Plotting the Result
fig, ax = plt.subplots()
ax.plot(sol.t,ca,sol.t,cb,sol.t,cc,sol.t,cd)
ax.set_xlabel('zeit in sekunden und so')
ax.set_ylabel('Concentration c in mol/m^3')
ax.legend(['H2','CO2','CH4','H2O'])
plt.show()

breakbreak = True

c = [ca,cb,cc,cd]
