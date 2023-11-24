import numpy
import Parameter_PDE_CH4
from Parameter_PDE_CH4 import update_dependent_values
import reaction_rate
from reaction_rate import reaction_rate

dc_i /dt = (1/tau) (c_i_in - ci) + nu_i * rho * r


r = reaction_rate(T,x,p)
update_dependent_values()
