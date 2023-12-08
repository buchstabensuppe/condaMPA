import sysidentpy as si
import numpy as np
import pandas as pd
from sysidentpy import metrics
from sysidentpy import narmax_base
from sklearn.model_selection import train_test_split
import CSTR1
from CSTR1 import simCSTR1
import numpy as np
import pandas as pd
from sysidentpy.metrics import root_mean_squared_error as rmse, mean_absolute_error as mae
from sysidentpy.metrics import mean_squared_error
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
from sysidentpy.utils.generate_data import get_siso_data
from catboost import CatBoostRegressor


seconds =int(5)
tstep = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 0]
x_train, y_train = simCSTR1(seconds, tstep, n_variables, x0s)
x_train_array = np.array(x_train)
y_train_array = np.array(y_train)

breakbreak = True

x_train_original, x_valid_original, y_train_original, y_valid_original = get_siso_data(
    n=1000,
    colored_noise=False,
    sigma=0.001,
    train_percentage=80
)

# Split data into training and testing sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train_array, y_train_array, test_size=0.2)
breakbreak = True


x1lag = list(range(1, 3))
x2lag = list(range(1, 3))
x3lag = list(range(1, 3))
y1lag = list(range(1, 3))
y2lag = list(range(1, 3))
y3lag = list(range(1, 3))

### another atempt, code provided by bard lol
# Define FROLS model structure
model = FROLS(ylag=[y3lag, y2lag, y1lag], xlag=[x1lag, x2lag, x3lag], basis_function=Polynomial(degree=2))

# Fit FROLS model
model = model.fit(X=x_train, y=y_train)

# Evaluate FROLS model performance
y_pred = model.predict(x_train)
rmse_value = rmse(y_train, y_pred)
mae_value = mae(y_train, y_pred)
print('RMSE:', rmse_value)
print('MAE:', mae_value)


breakbreak = True

###### Another atempt, using the code from: https://sysidentpy.org/examples/entropic_regression/
###### works, but error while fitting the model. Try to fix with @Alex
#
# import matplotlib.pyplot as plt
# from sysidentpy.model_structure_selection import ER
# from sysidentpy.basis_function._basis_function import Polynomial
# from sysidentpy.metrics import root_relative_squared_error
# from sysidentpy.utils.generate_data import get_siso_data
# from sysidentpy.utils.display_results import results
# from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
# from sysidentpy.residues.residues_correlation import (
#     compute_residues_autocorrelation,
#     compute_cross_correlation,
# )

#
# basis_function = Polynomial(degree=3)
#
# # model = ER(
# #     # ylag=8,
# #     # xlag=8,
# #     #ylag=[y1lag, y2lag, y3lag],
# #     ylag=[3],
# #     xlag=[x1lag, x2lag, x3lag],
# #     n_perm=3,
# #     k=3,
# #     skip_forward=True,
# #     estimator="recursive_least_squares",
# #     basis_function=basis_function,
# # )
#
# model.fit(X=x_train, y=y_train)
# yhat = model.predict(X=x_valid, y=y_valid)
# rrse = root_relative_squared_error(y_valid, yhat)
# print(rrse)
#
# r = pd.DataFrame(
#     results(
#         model.final_model,
#         model.theta,
#         model.err,
#         model.n_terms,
#         err_precision=8,
#         dtype="sci",
#     ),
#     columns=["Regressors", "Parameters", "ERR"],
# )
# print(r)
#
# plot_results(y=y_valid, yhat=yhat, n=1000)
# ee = compute_residues_autocorrelation(y_valid, yhat)
# plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
# x1e = compute_cross_correlation(y_valid, yhat, x_valid)
# plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

breakbreak = True











basis_function = Polynomial(degree=2)



#Define NARMAX model structure
# model = FROLS(order_selection=True,
#     n_info_values=10,
#     extended_least_squares=True,
#     # ylag=([[1, 3], [1, 3], [1, 3]]),
#     # xlag=([[1, 3], [1, 3], [1, 3]]),
#     ylag=[[1, 3],[1, 3],[1, 3]],
#     xlag=[x1lag, x2lag, x3lag],
#     info_criteria='aic',
#     estimator='Least_Squares()',
#     basis_function=basis_function)
#
# model.fit(X=X_train, y=y_train)


breakbreak = True

############ Same as other code, working but wont fit cause of different lenght of vectors, no Idea why
############ zusammen ansehen mit @Alex

from sysidentpy.general_estimators import NARX
catboost_narx = NARX(
    base_estimator=CatBoostRegressor(iterations=300, learning_rate=0.1, depth=8),
    #ylag=[y1lag, y2lag, y3lag],
    ylag=3,
    xlag=[x1lag, x2lag, x3lag],
    basis_function=basis_function,
    model_type="NARMAX",
    fit_params={"verbose": False},
)
catboost_narx.fit(X=x_train, y=y_train)
yhat = catboost_narx.predict(X=x_valid, y=y_valid, steps_ahead=1)
print("MSE: ", mean_squared_error(y_valid, yhat))
plot_results(y=y_valid, yhat=yhat, n=200)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

breakbreak = True


# Fit NARMAX model using MLP algorithm
model = model.fit(X=x_train, y=y_train)

# Evaluate model performance on testing data
y_pred = model.predict(x_test)

# Calculate RMSE and MAE for evaluation
rmse = rmse(y_test, y_pred)
mae = mae(y_test, y_pred)
print('RMSE:', rmse)
print('MAE:', mae)
# Load and prepare MIMO data
data = pd.read_csv('mimo_data.csv')

# Preprocess data (Normalization)
data_norm = (data - data.mean()) / data.std()

break_break = True
