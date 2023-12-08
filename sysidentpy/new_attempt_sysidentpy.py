import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
import CSTR1
from CSTR1 import simCSTR1
from sklearn.model_selection import train_test_split
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
# x_train, x_valid, y_train, y_valid = get_miso_data(
#     n=1000, colored_noise=False, sigma=0.001, train_percentage=90
# )
#
x_train_original, x_valid_original, y_train_original, y_valid_original = get_miso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)

seconds =int(5)
tstep = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 0]
x_train, y_train = simCSTR1(seconds, tstep, n_variables, x0s)

x_train, x_valid = train_test_split(x_train, test_size=0.2)
y_train, y_valid = train_test_split(y_train, test_size=0.2)

# print(x_train, x_test)
#
# u_test = x_test
# u_train = x_train

basis_function = Polynomial(degree=2)

model = FROLS(
    order_selection=True,
    n_terms=4,
    extended_least_squares=False,
    # ylag=[[1, 2], [1, 2], [1, 2]],
    # xlag=[[1, 2], [1, 2], [1, 2]],
    ylag = 2,
    xlag = [[1,2], [1,2]],
    info_criteria="aic",
    estimator="LeastSquares()",
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)

yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)

r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid[:, 0])
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")

breakbreak = True