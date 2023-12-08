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

# Dictionary of Parameters
p = {

    # Inlet Concentrations
    'cain': 10,
    'cbin': 0,
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
    'l': 50

}

# Vector of Initial Concentrations
p['cin'] = np.array([p['cain'], p['cbin'], p['ccin'], p['cdin']])


# Auxiliary function for Arrhenius-Calculation
def arrhenius(k0, ea, p):
    k = k0 * np.exp(- (ea / (p['r'] * p['temp'])))
    return k


# ODE-System
def ode_system(z, c, p):
    # Unpacking the Concentrations
    ca = c[0]
    cb = c[1]
    cc = c[2]
    cd = c[3]

    # Calculation of the rate constant
    k1 = arrhenius(p['k01'], p['ea1'], p)
    k2 = arrhenius(p['k02'], p['ea2'], p)
    k3 = arrhenius(p['k03'], p['ea3'], p)

    # Calculation of the rates
    r1 = k1 * np.power(ca, p['n1'])
    r2 = k2 * np.power(cb, p['n2'])
    r3 = k3 * np.power(cb, p['n3'])

    # Calculation of the gradients
    dcadz = -(p['a'] / p['q']) * r1
    dcbdz = (p['a'] / p['q']) * (r1 - r2 - r3)
    dccdz = (p['a'] / p['q']) * r2
    dcddz = (p['a'] / p['q']) * r3

    # Merging the Concentrations to one array
    dcdz = np.array([dcadz, dcbdz, dccdz, dcddz])

    return dcdz

# Solve the ODE

# Integration area
zspan = (0, p['l'])

# Define lambda-function
func = lambda z,c : ode_system(z,c,p)

# Solving the ODE-System for Initial Conditions
sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45',t_eval=np.linspace(0,p['l'],100))

# Unpacking the Trajectories of the Concentrations
ca = sol.y[0,:]
cb = sol.y[1,:]
cc = sol.y[2,:]
cd = sol.y[3,:]

# Plotting the Result
fig, ax = plt.subplots()
ax.plot(sol.t,ca,sol.t,cb,sol.t,cc,sol.t,cd)
ax.set_xlabel('Reactor coordinate z in m')
ax.set_ylabel('Concentration c in mol/m^3')
ax.legend(['A','B','C','D'])


asfasdfsadf