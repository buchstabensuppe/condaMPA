import sysidentpy as syp
import numpy as np
import pandas
import CSTRsysident
from CSTRsysident import simCSTR1 as sim
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)


#Daten erzeugen

seconds =int(5)
tstep = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 0]
data = sim(seconds, tstep, n_variables, x0s)

#Anwendung Sysidentpy

narmax = PolynomialNarmax(
    non_degree=2,  # The nonlinearity degree
    order_selection=True,  # Use order selection
    n_info_values=10,  # Number of information criteria to consider
    extended_least_squares=False,  # Use extended least squares
    ylag=1,  # The maximum number of lags for the output variable
    xlag=1,  # The maximum number of lags for the input variable
    estimator="least_squares"  # Use least squares estimator
)

poly = Polynomial(degree=2, include_bias=False)
fourier = Fourier(fourier_degree=[2, 2, 2], trigonometric_type="complex")
basis_functions = [poly, fourier]
narmax.basis_functions = basis_functions

identified_model = narmax.fit(X, Y)

print(identified_model)
