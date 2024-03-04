import sympy as sp  # Import SymPy for symbolic manipulation
import numpy as np
from scipy import integrate
from Parameter_PDE_CH4 import data
import matplotlib.pyplot as plt


def ode_as_strings(coeffs):
    # Define symbolic variables for states
    x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")  # Adjust variable names if needed

    # Initialize empty lists to store the ODEs symbolically
    ode_list_symbolic = []

    # Loop through each state (assuming 4 states) and build the ODEs
    for i in range(0, 4):  # Adjust the range based on the number of states
        # Extract coefficients for the current state
        state_coeffs = coeffs[i]

        # Use list comprehension to construct the symbolic expression for the derivative
        symbolic_derivative = sp.Add(*[coeff * sp.Symbol(f"f{j}") for j, coeff in enumerate(state_coeffs)])

        # Append the symbolic expression to the list
        ode_list_symbolic.append(symbolic_derivative)

    # If you prefer string representations for plotting or further manipulation:
    ode_list_strings = [sp.N(expr) for expr in ode_list_symbolic]


    return ode_list_symbolic
def simulate_sindy_result(coeffs, x0_sim, seconds, steps):
    ode_list_symbolic = ode_as_strings(coeffs)
    x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")  # Adjust variable names if needed

    print(ode_list_symbolic)
    def ode_system(t, x):
        # Replace placeholders with the actual symbolic expressions or strings
        dx1_dt = ode_list_symbolic[0].subs({x0: x[0], x1: x[1], x2: x[2], x3: x[3]})
        dx2_dt = ode_list_symbolic[1].subs({x0: x[0], x1: x[1], x2: x[2], x3: x[3]})
        dx3_dt = ode_list_symbolic[2].subs({x0: x[0], x1: x[1], x2: x[2], x3: x[3]})
        dx4_dt = ode_list_symbolic[3].subs({x0: x[0], x1: x[1], x2: x[2], x3: x[3]})
        dcdz = np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])

        return dcdz

    c_in  =  x0_sim * data['p_R'] / (data['R'] * data['T_gas_in'])

    zspan = (0, seconds)

    # Define lambda-function
    func = lambda t, c: ode_system(t, c)

    # Solving the ODE-System for Initial Conditions
    # sol = integrate.solve_ivp(func, zspan, p['cin'], method='RK45')#,t_eval=np.linspace(0, data['L_R'], 10))
    sol = integrate.solve_ivp(func, zspan, c_in, method='RK45', t_eval=np.linspace(0, seconds, steps))

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

