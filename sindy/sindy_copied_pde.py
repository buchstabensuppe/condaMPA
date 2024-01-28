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
'seconds': 1,
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
    c_in = data['x_in'] * data['p_R'] / (data['R'] * data['T_gas_in'])

    r = reaction_rate(p['T_gas_in'], x, 2E5)

    #r1 = k1 * np.power(ca, p['n1'])
    #r2 = k2 * np.power(cb, p['n2'])
    #r3 = k3 * np.power(cb, p['n3'])

    # Calculation of the gradients

    #dc_i / dt = (1 / tau)(c_i_in - ci) + nu_i * rho * r
    #tau = tau = Volumen/Volumenstrom = Querschnitt * Länge/ Volumenstrom

    ########### @Alex, breauchst du für das Volumen nicht pi? dafuq

    #tau = data['V_r']*data['F_in']
    V_r = 2 / 4 * np.pi * 0.01 ** 2
    tau = V_r * (6 * 1e-3 / 60 * (1.013E5 / 2E5) * (270+273.15 / 273.15))
    #tau = Volumen / Volumenstrom = Querschnitt * Länge / Volumenstrom
    #Volumen:
    #V_r = data["L_R"](2m) / 4 * np.pi * data["D_R"](0.01) ** 2
    V_r = p['L_r'] / 4 * np.pi * p['D_r'] ** 2;
    #Volumenstrom:
    F_in = (data["F_in_Nl_min"] * data["L_to_m3"] / data["min_to_s"]
                    * (1.013E5 / data["p_R"]) * (data["T_gas_in"] / 273.15))
    tau =V_r/F_in
    #print(tau)
    ########## cin fehlt, muss also berechnet werden aus Paramter Funktion ->
    #           ci = xi * summe(c)
    #           ci = ni/V
    #n_in_total = (data["p_R"] * data["V_r"]
    #                      / data["R"] / data["T_gas_in"])
    n_in_total = (p["p_R"] * V_r / p['R'] / p['T_gas_in'])
    x_in = np.array([0.8, 0.2, 0, 0])
    n_in = n_in_total * x_in


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
    # soll nu ein ny sein?
    ny = np.array([-1, -4, 2, 2])

    # print('tau:', tau)
    # print('c_in(0)', c_in[0])
    # print('ca:', ca)
    # print('ny(0):', ny[0])
    # print('r:', r)

    #tau = 0.2
    # dcadz = (1 / tau)*(c_in[0] - ca) + ny[0] * 1032 * r
    # dcbdz = (1 / tau)*(c_in[1] - cb) + ny[1] * 1032 * r
    # dccdz = (1 / tau)*(c_in[2] - cc) + ny[2] * 1032 * r #data['rho_cat'] * r
    # dcddz = (1 / tau)*(c_in[3] - cd) + ny[3] * 1032 * r

    x0 = ny[0]
    x1 = ny[1]
    x2 = ny[2]
    x3 = ny[3]

    # weak sindy result for
    # dcadz = 1240.969 * 1 + -144.725 * x0 + -194.589 * x1 + 334.655 * x2 + 173.464 * x3 + 3.156 * x0 * x0 + -2.821 * x1 * x1 + -7.745 * x2 * x2 + 3.312 * x3 * x3 + 6.342 * x0 * x1 + -8.013 * x0 * x2 + -6.082 * x0 * x3 + -2.073 * x1 * x2 + -0.940 * x1 * x3 + -0.245 * x2 * x3
    # dcbdz = 2403.040 * 1 + -137.797 * x0 + 6.226 * x1 + -81.575 * x2 + 43.940 * x3 + 1.870 * x0 * x0 + 4.226 * x1 * x1 + -0.221 * x2 * x2 + -1.371 * x3 * x3 + -0.342 * x0 * x1 + 3.035 * x0 * x2 + -0.512 * x0 * x3 + 0.807 * x1 * x2 + 0.479 * x1 * x3 + -3.253 * x2 * x3
    # dccdz = 1601.529 * 1 + -89.487 * x0 + 129.126 * x1 + 32.148 * x2 + -91.630 * x3 + 1.153 * x0 * x0 + 2.232 * x1 * x1 + -1.854 * x2 * x2 + 1.244 * x3 * x3 + -3.309 * x0 * x1 + 0.078 * x0 * x2 + 3.100 * x0 * x3 + -4.223 * x1 * x2 + -0.165 * x1 * x3 + -4.795 * x2 * x3
    # dcddz = -3495.433 * 1 + 209.507 * x0 + 72.447 * x1 + -1.508 * x2 + 5.710 * x3 + -3.087 * x0 * x0 + -2.497 * x1 * x1 + 0.263 * x2 * x2 + 2.601 * x3 * x3 + -1.108 * x0 * x1 + -0.126 * x0 * x2 + -0.821 * x0 * x3 + -2.872 * x1 * x2 + -5.129 * x1 * x3 + 0.362 * x2 * x3

    dcadz = 2493.614 * 1 + -84.787 * x0 + 58.666 * x1 + -12.306 * x2 + 0.682 * x0 * x0 + 0.919 * x1 * x1 + 0.013 * x2 * x2 + -1.233 * x0 * x1 + 0.267 * x0 * x2 + -0.129 * x1 * x2
    dcbdz = -1003.706 * 1 + 29.517 * x0 + -19.411 * x1 + 9.503 * x2 + -0.188 * x0 * x0 + -0.003 * x1 * x1 + 0.002 * x2 * x2 + 0.175 * x0 * x1 + -0.193 * x0 * x2 + 0.187 * x1 * x2
    dccdz = -2984.342 * 1 + 93.189 * x0 + -56.606 * x1 + 21.710 * x2 + -0.645 * x0 * x0 + -0.494 * x1 * x1 + -0.004 * x2 * x2 + 0.866 * x0 * x1 + -0.472 * x0 * x2 + 0.370 * x1 * x2
    dcddz = 0
    # weak sindy result for

    # dcadz = -(p['a'] / p['q']) * r1
    # dcbdz = (p['a'] / p['q']) * (r1 - r2 - r3)
    # dccdz = (p['a'] / p['q']) * r2
    # dcddz = (p['a'] / p['q']) * r3

    # Merging the Concentrations to one array
    dcdz = np.array([dcadz, dcbdz, dccdz, dcddz])

    return dcdz

# Solve the ODE


# Integration area
zspan = (0, p['seconds'])


# Define lambda-function
func = lambda t,c : ode_system(t,c,p)


# Solving the ODE-System for Initial Conditions
#sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45')#,t_eval=np.linspace(0, data['L_R'], 10))
#sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45',t_eval=np.linspace(0, p['seconds'], 1000))

zspan = (0, 50)
sol = integrate.solve_ivp(func, zspan, [100, 50, 10, 0], method='RK45',t_eval=np.linspace(0, 50, 1000))

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
