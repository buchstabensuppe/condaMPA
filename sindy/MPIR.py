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


import numpy
import Parameter_PDE_CH4
from Parameter_PDE_CH4 import update_dependent_values
import reaction_rate
from reaction_rate import reaction_rate
from Parameter_PDE_CH4 import data


print(data['x_in'])


#dc_i /dt = (1/tau) (c_i_in - ci) + nu_i * rho * r

# T_0 = data['T_gas_in']
# print(T_0)

r_0 = reaction_rate(data['T_gas_in'],data['X_0'],data['p_R'])
print(r_0)
#update_dependent_values()
