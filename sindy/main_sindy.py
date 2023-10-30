import CSTR1
from CSTR1 import simCSTR1
import numpy as np
import pysindy as ps
#CSTR Simulation aufrufen und Simulationsdaten importeren
seconds =int(5) #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
tstep = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 0]
data = simCSTR1(seconds, tstep, n_variables, x0s)

############ sindy ################
t = np.linspace(0, 1, 5000)

feature_names=["a", "b", "c"]
#
# opt = pysindy.STLSQ(threshold=0.004)
# sindy_opt = ps.SINDyPI(
#     threshold=1e-6,
#     tol=1e-8,
#     thresholder="l1",
#     max_iter=20000,
#)
#
# sindy_library = ps.SINDyPILibrary(
#     library_functions=library_functions,
#     x_dot_library_functions=x_dot_library_functions,
#     t=t,
#     function_names=library_function_names,
#     include_bias=True,
# )
# opt = ps.SR3(
#     threshold=0.04, thresholder="l1", max_iter=1000000, normalize_columns=True, tol=1e-6
# )
#opt = ps.SINDy()
#opt = ps.BaseOptimizer(0)
model = ps.SINDy(feature_names=feature_names, optimizer=opt)


model.fit(data, t=t)
model.print()
############## calling sindy (out of different .py) with multiple trajectories ##########
#
# ssr_optimizer = ps.SSR()
# model = ps.SINDy(optimizer=ssr_optimizer)


#
# ############ sindy threshold scan ################
# threshold_scan = np.linspace(0,1,11)
# coefs = []
#
# rmse = mean_square_error(x_train,
#                          np.zeros(x_train.shape,),
#                          squared = False)
#
# for i, threshold in enumerate(threshold_scan):
#     opt = ps.STLSQ(threshold=threshold)
#     model = ps.SINDy(feature_names=feature_names, optimizer=opt)
#     model.fit()


## sindy example ##
#
# t = np.linspace(0, 1, 100)
# x = 3 * np.exp(-2 * t)
# y = 0.5 * np.exp(t)
# X = np.stack((x, y), axis=-1)  # First column is x, second is y
# model = ps.SINDy(feature_names=["x", "y"])
# model.fit(X, t=t)
# model.print()

simtest(seconds, tstep, n_variables)
