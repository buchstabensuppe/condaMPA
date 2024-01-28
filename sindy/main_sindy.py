import CSTR1
from CSTR1 import simCSTR1
import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
#CSTR Simulation aufrufen und Simulationsdaten importeren
import matplotlib.pyplot as plt


seconds =5 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.001 #[s]
n_variables = 3
x0s = [100, 50, 10]
data = simCSTR1(seconds, dt, n_variables, x0s)

# mein CSTR in original namen übersetzen

t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])
u_train = data
u_train_clean = u_train
# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# Instantiate and fit a non-weak SINDy model
ode_lib = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    include_bias=True,
)
optimizer = ps.SR3(
    threshold=100, thresholder="l0", max_iter=10000, normalize_columns=True, tol=1e-10
)
original_model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
original_model.fit(u_train, t=dt, quiet=True)
print("Regular model: ")
original_model.print()
err_not_weak = np.sqrt(
    (np.sum((u_dot - optimizer.Theta_ @ optimizer.coef_.T) ** 2) / np.sum(u_dot**2))
    / u_dot.shape[0]
)

u_pred = original_model.simulate(x0s, t=t_train, integrator="odeint")
# um u_dot zu predicten muss das Anfangswertproblem für das original_model gelöst werden. Eventuell mit solve_ivp von https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html
# u_dot_pred = original_model.predict(u_test)
print(u_pred)


#jetzt einmal mit noise, dann nochmal ausgeben und über erstes Ergebnis plotten

#### noise auf die Daten packen:
u_test = np.copy(u_train)
rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
u_clean = u_test
u_train = u_train + np.random.normal(0, rmse / 80.0, u_train.shape)  # Add 20% noise
print('u_train with noise:', u_train)
rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
u_test = u_test + np.random.normal(0, rmse / 80.0, u_test.shape)  # Add 20% noise
u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)


# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# Instantiate and fit a non-weak SINDy model
ode_lib = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    include_bias=True,
)
optimizer = ps.SR3(
    threshold=100, thresholder="l0", max_iter=10000, normalize_columns=True, tol=1e-10
)
model_verrauscht = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model_verrauscht.fit(u_train, t=dt, quiet=True)
print("Regular model: ")
model_verrauscht.print()
err_not_weak = np.sqrt(
    (np.sum((u_dot - optimizer.Theta_ @ optimizer.coef_.T) ** 2) / np.sum(u_dot**2))
    / u_dot.shape[0]
)

u_pred_rauschen = model_verrauscht.simulate(x0s, t=t_train, integrator="odeint")
print(u_pred_rauschen)
# Plot the ground truth, weak form, and non-weak trajectory predictions
feature_names = ["x", "y", "z"]
u_weak = original_model.simulate(x0s, t=t_train, integrator="odeint")


plt.figure(figsize=(16, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(t_train, u_train_clean[:, i], "k", label=r"$q$ without added noise")
    plt.plot(t_train, u_test[:, i], "c", label=r"$q$ with added noise")
    plt.plot(t_train, u_pred[:, i], "r", label=r"$q$ prediction")
    plt.plot(t_train, u_pred_rauschen[:, i], "b", label=r"$q$ weak form prediction")
    plt.grid(True)
    plt.ylabel(feature_names[i], fontsize=14)
    plt.xlabel("t", fontsize=14)
    plt.ylim(-100, 100)
    if i == 2:
        plt.legend()

plt.show()

breakbreak = 1