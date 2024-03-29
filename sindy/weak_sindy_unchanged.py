#source: https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html#Test-weak-form-ODE-functionality-on-Lorenz-equation
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from pysindy.utils.odes import lorenz
import CSTR1
from averaging_measurements_for_noise_reduction import average_noisy_measurement
from CSTR1 import simCSTR1
from MPIR_callable_function import MPI_reactor
import pysindy as ps

# generating data:
seconds = 2 #[s]  <- WARNING if you change these numbers, also num at t=np.linspace has to be changed
dt = 0.002 #[s]
dt_time_seconds = int(seconds/dt)
filter = False

# Simulation auswählen: basic batch reactor = 1, MPI CSTR = 2, 3 = MPI CSTR but average of multiple measurements
reactor_choice = 2
if reactor_choice == 1:
    n_variables = 3
    x0s = [100, 50, 10]
    x0 = x0s
    data = simCSTR1(seconds, dt, n_variables, x0s)
    u_train = data

if reactor_choice == 2:
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.6, 0.15, 0, 0])
    data_raw = MPI_reactor(seconds, dt_time_seconds, x0, dont_plot = True)
    data_raw_test = MPI_reactor(seconds, dt_time_seconds, x0_test, dont_plot = True)
    x0s = data_raw[0]
    x0s_test = data_raw_test[0]
    data_tmp = np.array(data_raw)
    data_tmp_test = np.array(data_raw_test)
    data = np.zeros((dt_time_seconds, 4))
    data_test = np.zeros((dt_time_seconds,4))
    for i in range(dt_time_seconds):
        data[i] = [data_tmp[0, i], data_tmp[1, i], data_tmp[2, i], data_tmp[3, i]]
        data_test[i] = [data_tmp_test[0, i], data_tmp_test[1, i], data_tmp_test[2, i], data_tmp_test[3, i]]
    u_train = data
    u_test = data_test

if reactor_choice == 3:
    average_of_n = 100
    n_variables = 4
    x0 = np.array([0.8, 0.2, 0, 0])
    x0_test = np.array([0.6, 0.15, 0, 0])
    u_dot_clean, u_clean, u_train, u_test, u_dot = average_noisy_measurement(average_of_n, seconds, dt_time_seconds, x0)

t_train = np.arange(0, seconds, dt)
t_train_span = (t_train[0], t_train[-1])


# applying noise
if reactor_choice != 3:
    rmse = mean_squared_error(u_train, np.zeros((u_train).shape), squared=False)
    u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
    u_clean = u_test
    u_train = u_train + np.random.normal(0, rmse / 160.0, u_train.shape)  # Add 20% noise
    rmse = mean_squared_error(u_test, np.zeros(u_test.shape), squared=False)
    u_test = u_test + np.random.normal(0, rmse / 160.0, u_test.shape)  # Add 20% noise
    u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)

# setting negative values to zero: a *= (a>0)
u_train *= (u_train > 0)
u_test *= (u_test > 0)
u_dot *= (u_dot > 0)

# applying noise filters: (none of which are working <sad pepe face>)
if filter == True:
    from scipy.signal import medfilt
    # Example usage
    u_train_filtered = medfilt(u_train, kernel_size=5)
    u_test_filtered = medfilt(u_test, kernel_size=5)
    u_train = u_train_filtered
    u_test = u_test_filtered

library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

#library set ending without sin cos tan
# library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y, lambda x, y: x * x *y, lambda x: x * x * x]
# library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y, lambda x, y: x + x + y, lambda x: x + x + x]

# library set ending with xtanx
# library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y, lambda x, y: x * x *y, lambda x: x * x * x,
#                      lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.tan(x),
#                      lambda x: x*np.sin(x), lambda x: x*np.cos(x), lambda x: x*np.tan(x),
#                      lambda x,y: y*np.sin(x), lambda x,y: y*np.cos(x), lambda x,y: y*np.tan(x)]
# library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y, lambda x, y: x + x + y, lambda x: x + x + x,
#                           lambda x: "sin"+x, lambda x: "cos"+x, lambda x: "tan"+x,
#                           lambda x: x+"sin"+x, lambda x: x+"cos"+x, lambda x: x+"tan"+x,
#                           lambda x,y: y+"sin"+x, lambda x,y: y+"cos"+x, lambda x,y: y+"tan"+x]

# hyperparameter tuning settings:

thresholder_iterated = 0.00001
thresholder_variation = 0.00001
results = np.ones((100,2))
n_iterations = 1000
simulation_timeout = 60  # (s)
mse_tracker = 0
mse_tracker_steps = 0
results_mse = []
results_thresholder = []
# managing timeout for simulation of results:
import signal
import time


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()


def loop(n):
    for sec in range(n):
        print("sec {}".format(sec))
        time.sleep(1)

x_train_multi = []
x_train_multi.append(u_train)
x_train_multi.append(u_test)
for ji in range(n_iterations):
    thresholder_iterated += thresholder_variation
    # Scan over the number of integration points and the number of subdomains
    n = 10
    errs = np.zeros((n))
    K_scan = np.linspace(20, 2000, n, dtype=int) #ursprünglich: stop = 2000, höhere Werte liefern bessere Ergebnisse
    for i, K in enumerate(K_scan):
        pde_lib = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=t_train,
            include_bias=True,
            is_uniform=True,
            K=2,
        )
        #opt = ps.STLSQ(threshold=0.05, alpha=1e-5, normalize_columns=True)

        opt = ps.SR3(
            threshold=thresholder_iterated,  #Standard war 0.05, appearently deutlich bessere Ergebnisse mit geringerem Threshold
            thresholder="l1",
            max_iter=10000000000,
            normalize_columns=True,
            tol=1e-4,
        )
        u_dot_train_integral = pde_lib.convert_u_dot_integral(u_train)
        u_valid_train_integral = pde_lib.convert_u_dot_integral(u_test)

        # Instantiate and fit the SINDy model with the integral of u_dot
        model = ps.SINDy(feature_library=pde_lib, optimizer=opt)
        model.fit(x_train_multi, quiet=True, multiple_trajectories=True)
        validate_with_test_data = False
        # if validate_with_test_data == False:
        #     errs[i] = np.sqrt(
        #         (
        #             np.sum((u_valid_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
        #             / np.sum(u_valid_train_integral ** 2)
        #         )
        #         / u_valid_train_integral.shape[0]
        #     )
        # else:
        #     errs[i] = np.sqrt(
        #         (
        #             np.sum((u_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
        #             / np.sum(u_dot_train_integral ** 2)
        #         )
        #         / u_dot_train_integral.shape[0]
        #     )
        model.print()

    #ploting error graph
    if False:
        plt.title('Convergence of weak SINDy, hyperparameter scan', fontsize=12)
        plt.plot(K_scan,errs)
        plt.xlabel('Number of subdomains', fontsize=16)
        plt.ylabel('Error', fontsize=16)
        plt.show()

    #simulation of discovered system, with timeout:
    from symbolic_model_to_simulation_2 import simulate_sindy_result_3
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(simulation_timeout)
    timed_out = False
    try:
        results_sindy_simulation = simulate_sindy_result_3(model.coefficients(), x0, seconds, dt_time_seconds, plot_results = False, e3set = False)
        results_sindy_simulation_test_set = simulate_sindy_result_3(model.coefficients(), x0_test, seconds, dt_time_seconds, plot_results = False, e3set = False)
    except TimeOutException as ex:
        print(ex)
        timed_out = True

    #results_sindy_simulation = simulate_sindy_result_2(model.coefficients(), x0, seconds, dt_time_seconds, plot_results = False)

    # plotting comparison for each variable, without noise:
    #
    # # Extract inner lists from dictionaries
    # list1_inner = data_raw
    # list2_inner = results_sindy_simulation
    #
    # # Define labels for the plots
    # label1 = "CSTR Model"
    # label2 = "Resulting Sindy PDE"
    #
    # # Create subplots for 4 graphs
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # Adjust figsize as needed
    #
    # # Iterate through each subplot and plot data from separate lists
    # for i in range(2):
    #     for j in range(2):
    #         index = i * 2 + j  # Calculate index for each subplot (0 to 3)
    #
    #         # Plot data from list1 (inner list at index)
    #         axes[i, j].plot(list1_inner[index], label=label1)
    #
    #         # Plot data from list2 (inner list at index)
    #         axes[i, j].plot(list2_inner[index], label=label2)
    #
    #         axes[i, j].set_xlabel("Index")
    #         axes[i, j].set_ylabel("Value")
    #         axes[i, j].set_title(f"Variable {index + 1} Comparison")  # Add subplot title
    #         axes[i, j].legend()
    #
    # # Adjust layout and display the plot
    # plt.tight_layout()
    # plt.show()

    # plotting comparison with noise:
    #list1_inner = data_raw
    # list2_inner = results_sindy_simulation
    # list3_inner = u_train.T  # Sample array
    #
    # # Define labels for the plots
    # labels = ["CSTR Model", "PDE calculated with weak sindy", "CSTR Model + noise"]
    #
    # # Create subplots for 3 graphs (2x2 grid)
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Adjust figsize as needed
    #
    # # Function to plot data in a subplot
    # def plot_data(data_list, subplot, label, alpha=1):  # Add alpha parameter with default 1
    #   line, = subplot.plot(data_list, label=label, alpha=alpha)  # Unpack line object
    #   return line  # Return the line object
    #
    # # Iterate through subplots and plot data from each list
    # for i in range(2):
    #   for j in range(min(2, len(labels))):
    #     if j >= len(labels):
    #       break
    #     # Access data for the current subplot
    #     current_list1 = list1_inner[i]
    #     current_list2 = list2_inner[i]
    #     current_list3 = list3_inner[i]
    #
    #     # Plot data in the current subplot (set alpha for list3)
    #     line1 = plot_data(current_list1, axes[i, j], labels[0])  # Use labels[0] for all lines
    #     line2 = plot_data(current_list2, axes[i, j], labels[1])
    #     line3 = plot_data(current_list3, axes[i, j], labels[2], alpha=0.5)  # Set alpha to 0.5 for transparency
    #
    #     # Create a legend for the current subplot with all three labels
    #     axes[i, j].legend([line1, line2, line3], labels)  # Provide line objects and labels directly
    #
    # # Adjust layout and display the plot
    # plt.tight_layout()
    # plt.show()

    # # plotting comparison with noise, including test set
    # list1_inner = u_test.T
    # list2_inner = results_sindy_simulation
    # list3_inner = u_train.T  # Sample array
    # list4_inner = results_sindy_simulation_test_set
    # print(list2_inner - list3_inner)
    # # Define labels for the plots
    # labels = ["CSTR (validation data) + noise", "resulting PDE, IC = validation IC ", "CSTR (training data) + noise", "resulting PDE, IC = test IC"]
    #
    # # Create subplots for 4 graphs (2x2 grid) - adjust figsize as needed
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    #
    # # Function to plot data in a subplot
    # def plot_data(data_list, subplot, label, alpha=1):  # Add alpha parameter with default 1
    #   line, = subplot.plot(data_list, label=label, alpha=alpha)  # Unpack line object
    #   return line  # Return the line object
    #
    # # Iterate through subplots and plot data from each list (outer loop only once)
    # for i in range(2):  # Loop through rows only once
    #   for j in range(2):  # Loop through columns
    #     if j >= len(labels):  # Check if index exceeds label count (not needed here)
    #       break
    #     # Access data for the current subplot
    #     if i == 0:
    #         current_list1 = list1_inner[j]
    #         current_list2 = list2_inner[j]
    #         current_list3 = list3_inner[j]
    #         current_list4 = list4_inner[j]
    #     if i == 1:
    #         current_list1 = list1_inner[j+2]
    #         current_list2 = list2_inner[j+2]
    #         current_list3 = list3_inner[j+2]
    #         current_list4 = list4_inner[j+2]
    #
    #
    #     # Plot data in the current subplot (set alpha for list3)
    #     line1 = plot_data(current_list1, axes[i, j], labels[0], alpha=0.5)
    #     line2 = plot_data(current_list2, axes[i, j], labels[1])
    #     line3 = plot_data(current_list3, axes[i, j], labels[2], alpha=0.5)  # Set alpha to 0.5 for transparency
    #     line4 = plot_data(current_list4, axes[i, j], labels[3])
    #
    #     # Create a legend for the current subplot with all three labels
    #     axes[i, j].legend([line1, line2, line3, line4], labels)  # Provide line objects and labels directly
    #
    # # Adjust layout and display the plot
    # plt.tight_layout()
    # plt.show()

    # saving data for hyperparameter plot:
    if not timed_out:
        # Assuming u_train.T has shape (4, 2500) and results_sindy_simulation has shape (4, 860)
        shorter_list = results_sindy_simulation
        longer_list = u_train.T
        shorter_list_2 = results_sindy_simulation_test_set
        longer_list_2 = u_test.T


        # Use the minimum length of the lists to slice the longer list
        common_length = min(len(shorter_list[0]), len(longer_list[0]))
        sliced_longer_list = longer_list[:, :common_length]

        common_length_2 = min(len(shorter_list_2[0]), len(longer_list_2[0]))
        sliced_longer_list_2 = longer_list_2[:, :common_length_2]

        # Now you can safely subtract
        squared_differences = np.square(np.subtract(sliced_longer_list, shorter_list))
        squared_differences_2 = np.square(np.subtract(sliced_longer_list_2, shorter_list_2))

        # # Calculate the mean squared error (MSE)
        # squared_differences = np.square(np.subtract(u_train.T, results_sindy_simulation))
        mse = np.mean(squared_differences)
        mse_2 = np.mean(squared_differences_2)
        print('mse:', mse)
        print('mse_2:', mse_2)

        mse_tracker_steps += 1
        mse_tracker += mse
        if mse_tracker_steps == 10:
            print(mse_tracker/10)
            mse_tracker = 0
            mse_tracker_steps = 0
        if mse == 'inf':
            mse = 999
        if mse < 1 and mse_2 < 1:

            # plotting comparison with noise, including test set
            list1_inner = u_test.T
            list2_inner = results_sindy_simulation
            list3_inner = u_train.T  # Sample array
            list4_inner = results_sindy_simulation_test_set
            print(list2_inner - list3_inner)
            # Define labels for the plots
            labels = ["CSTR (validation data) + noise", "resulting PDE, IC = validation IC ",
                      "CSTR (training data) + noise", "resulting PDE, IC = test IC"]

            # Create subplots for 4 graphs (2x2 grid) - adjust figsize as needed
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))


            # Function to plot data in a subplot
            def plot_data(data_list, subplot, label, alpha=1):  # Add alpha parameter with default 1
                line, = subplot.plot(data_list, label=label, alpha=alpha)  # Unpack line object
                return line  # Return the line object


            # Iterate through subplots and plot data from each list (outer loop only once)
            for i in range(2):  # Loop through rows only once
                for j in range(2):  # Loop through columns
                    if j >= len(labels):  # Check if index exceeds label count (not needed here)
                        break
                    # Access data for the current subplot
                    if i == 0:
                        current_list1 = list1_inner[j]
                        current_list2 = list2_inner[j]
                        current_list3 = list3_inner[j]
                        current_list4 = list4_inner[j]
                    if i == 1:
                        current_list1 = list1_inner[j + 2]
                        current_list2 = list2_inner[j + 2]
                        current_list3 = list3_inner[j + 2]
                        current_list4 = list4_inner[j + 2]

                    # Plot data in the current subplot (set alpha for list3)
                    line1 = plot_data(current_list1, axes[i, j], labels[0], alpha=0.5)
                    line2 = plot_data(current_list2, axes[i, j], labels[1])
                    line3 = plot_data(current_list3, axes[i, j], labels[2],
                                      alpha=0.5)  # Set alpha to 0.5 for transparency
                    line4 = plot_data(current_list4, axes[i, j], labels[3])

                    # Create a legend for the current subplot with all three labels
                    axes[i, j].legend([line1, line2, line3, line4], labels)  # Provide line objects and labels directly

            # Adjust layout and display the plot
            plt.tight_layout()
            plt.show()
        # Print the MSE
        #print("Mean Squared Error (MSE):", mse)

        results_mse.append(mse)
        results_thresholder.append(thresholder_iterated)
        print(results_mse, results_thresholder)
breakbreak = True

#plot of hyperparameter tuning results:

# plotting the points
plt.plot(results_thresholder, results_mse)

# naming the x axis
plt.xlabel('thresholder value')
# naming the y axis
plt.ylabel('mse')
plt.yscale("log")

# giving a title to my graph
plt.title('hyperparameter tuning: thresholder')

# function to show the plot
plt.show()

#TODO: Spacio Temporal Grid implementieren wie in Doku
#TODO: Fehler melden (erst Alex fragen): bei library entry x*x*y vergrößert sich model.coefficients() nur um 6, werden also nicht alle Möglichkeiten berücksichtigt
#TODO: Gut laufendes result mit anderen ICs laufen lassen
#TODO: Testen ob mit weniger noise bessere Ergebnisse. Standard die letzten Wochen war 80
#TODO: nach gespräch mit Alex: 1: ohne noise testen
                        #2: plots verbesern, sodass gleich auch das test set mit geplottet wird