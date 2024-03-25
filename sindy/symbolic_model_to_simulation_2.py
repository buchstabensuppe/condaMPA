import sympy as sp  # Import SymPy for symbolic manipulation
import numpy as np
from scipy import integrate
from Parameter_PDE_CH4 import data
import matplotlib.pyplot as plt


def ode_as_strings(coeffs):
    # Define symbolic variables for states
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = sp.symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14")  # Adjust variable names if needed
    variable_combinations = ["x0", "x1", "x2", "x3", "x0 * x0", "x1 * x1", "x2 * x2", "x3 * x3", "x0 * x1", "x0 * x2",
                             "x0 * x3", "x1* x2", "x1 * x3", "x2 * x3"]
    # Initialize empty lists to store the ODEs symbolically
    ode_list_symbolic = []

    # Loop through each state (assuming 4 states) and build the ODEs
    for i in range(0, 4):  # Adjust the range based on the number of states
        # Extract coefficients for the current state
        state_coeffs = coeffs[i]

        # Use list comprehension to construct the symbolic expression for the derivative
        #symbolic_derivative = sp.Add(*[coeff * sp.Symbol(variable_combinations[j]) for j, coeff in enumerate(state_coeffs)])
        # Use list comprehension to construct the symbolic expression for the derivative
        symbolic_derivative = sp.Add(*[coeff * sp.Symbol(f"x{j}") for j, coeff in enumerate(state_coeffs)])

        # Append the symbolic expression to the list
        ode_list_symbolic.append(symbolic_derivative)

    # If you prefer string representations for plotting or further manipulation:
    ode_list_strings = [sp.N(expr) for expr in ode_list_symbolic]


    return ode_list_symbolic
def simulate_sindy_result_2(coeffs, x0_sim, seconds, steps, plot_results):
    ode_list_symbolic = ode_as_strings(coeffs)
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = sp.symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14")  # Adjust variable names if needed

    print(ode_list_symbolic)
    def ode_system(t, x):
        # Replace placeholders with the actual symbolic expressions or strings
        z = 1
        dx1_dt = sp.N(ode_list_symbolic[0].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3], x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3], x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3]}))
        dx2_dt = sp.N(ode_list_symbolic[1].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3], x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3], x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3]}))
        dx3_dt = sp.N(ode_list_symbolic[2].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3], x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3], x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3]}))
        dx4_dt = sp.N(ode_list_symbolic[3].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3], x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3], x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3]}))
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
    if plot_results:
        # Plotting the Result
        fig, ax = plt.subplots()
        ax.plot(sol.t,ca,sol.t,cb,sol.t,cc,sol.t,cd)
        ax.set_xlabel('zeit in sekunden und so')
        ax.set_ylabel('Concentration c in mol/m^3')
        ax.legend(['H2','CO2','CH4','H2O'])
        plt.title('automatic plot of sindy result:')
        plt.show()

    c = [ca, cb, cc, cd]
    return c

def simulate_sindy_result_3(coeffs, x0_sim, seconds, steps, plot_results, e3set):
    ode_list_symbolic = ode_as_strings(coeffs)
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66 = (
        sp.symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 x65 x66"))  # Adjust variable names if needed

    print(ode_list_symbolic)
    def ode_system(t, x):
        # Replace placeholders with the actual symbolic expressions or strings
        z = 1
        if not e3set:
            dx1_dt = sp.N(ode_list_symbolic[0].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2], x19: x[1] * x[1] * x[3],
                                                     x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0], x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]),
                                                     x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]),
                                                     x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3])}))
            dx2_dt = sp.N(ode_list_symbolic[1].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2], x19: x[1] * x[1] * x[3],
                                                     x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0], x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]),
                                                     x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]),
                                                     x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3])}))
            dx3_dt = sp.N(ode_list_symbolic[2].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2], x19: x[1] * x[1] * x[3],
                                                     x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0], x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]),
                                                     x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]),
                                                     x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3])}))
            dx4_dt = sp.N(ode_list_symbolic[3].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3], x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2], x19: x[1] * x[1] * x[3],
                                                     x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0], x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]),
                                                     x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]),
                                                     x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3])}))
            dcdz = np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])
        else:
            dx1_dt = sp.N(ode_list_symbolic[0].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3],
                                                     x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2],
                                                     x19: x[1] * x[1] * x[3], x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0],
                                                     x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]), x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]), x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3]),
                                                     x37: x[0]*np.sin(x[0]), x38: x[1]*np.sin(x[1]), x39: x[2]*np.sin(x[2]), x40: x[3]*np.sin(x[3]), x41: x[0]*np.cos(x[0]), x42: x[1]*np.cos(x[1]), x43: x[2]*np.cos(x[2]), x44: x[3]*np.cos(x[3]), x45: x[0]*np.tan(x[0]), x46: x[1]*np.tan(x[1]), x47: x[2]*np.tan(x[2]), x48: x[3]*np.tan(x[3]),
                                                     x49: x[1]*np.sin(x[0]), x50: x[2]*np.sin(x[0]), x51: x[3]*np.sin(x[0]), x52: x[2]*np.sin(x[1]), x53: x[3]*np.sin(x[1]), x54: x[3]*np.sin(x[2]),
                                                     x55: x[1] * np.cos(x[0]), x56: x[2] * np.cos(x[0]), x57: x[3] * np.cos(x[0]), x58: x[2] * np.cos(x[1]), x59: x[3] * np.cos(x[1]), x60: x[3] * np.cos(x[2]),
                                                     x61: x[1] * np.tan(x[0]), x62: x[2] * np.tan(x[0]), x63: x[3] * np.tan(x[0]), x64: x[2] * np.tan(x[1]), x65: x[3] * np.tan(x[1]), x66: x[3] * np.tan(x[2]),
                                                     }))
            dx2_dt = sp.N(ode_list_symbolic[1].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3],
                                                     x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2],
                                                     x19: x[1] * x[1] * x[3], x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0],
                                                     x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]), x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]), x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3]),
                                                     x37: x[0]*np.sin(x[0]), x38: x[1]*np.sin(x[1]), x39: x[2]*np.sin(x[2]), x40: x[3]*np.sin(x[3]), x41: x[0]*np.cos(x[0]), x42: x[1]*np.cos(x[1]), x43: x[2]*np.cos(x[2]), x44: x[3]*np.cos(x[3]), x45: x[0]*np.tan(x[0]), x46: x[1]*np.tan(x[1]), x47: x[2]*np.tan(x[2]), x48: x[3]*np.tan(x[3]),
                                                     x49: x[1]*np.sin(x[0]), x50: x[2]*np.sin(x[0]), x51: x[3]*np.sin(x[0]), x52: x[2]*np.sin(x[1]), x53: x[3]*np.sin(x[1]), x54: x[3]*np.sin(x[2]),
                                                     x55: x[1] * np.cos(x[0]), x56: x[2] * np.cos(x[0]), x57: x[3] * np.cos(x[0]), x58: x[2] * np.cos(x[1]), x59: x[3] * np.cos(x[1]), x60: x[3] * np.cos(x[2]),
                                                     x61: x[1] * np.tan(x[0]), x62: x[2] * np.tan(x[0]), x63: x[3] * np.tan(x[0]), x64: x[2] * np.tan(x[1]), x65: x[3] * np.tan(x[1]), x66: x[3] * np.tan(x[2]),
                                                     }))
            dx3_dt = sp.N(ode_list_symbolic[2].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3],
                                                     x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2],
                                                     x19: x[1] * x[1] * x[3], x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0],
                                                     x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]), x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]), x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3]),
                                                     x37: x[0]*np.sin(x[0]), x38: x[1]*np.sin(x[1]), x39: x[2]*np.sin(x[2]), x40: x[3]*np.sin(x[3]), x41: x[0]*np.cos(x[0]), x42: x[1]*np.cos(x[1]), x43: x[2]*np.cos(x[2]), x44: x[3]*np.cos(x[3]), x45: x[0]*np.tan(x[0]), x46: x[1]*np.tan(x[1]), x47: x[2]*np.tan(x[2]), x48: x[3]*np.tan(x[3]),
                                                     x49: x[1]*np.sin(x[0]), x50: x[2]*np.sin(x[0]), x51: x[3]*np.sin(x[0]), x52: x[2]*np.sin(x[1]), x53: x[3]*np.sin(x[1]), x54: x[3]*np.sin(x[2]),
                                                     x55: x[1] * np.cos(x[0]), x56: x[2] * np.cos(x[0]), x57: x[3] * np.cos(x[0]), x58: x[2] * np.cos(x[1]), x59: x[3] * np.cos(x[1]), x60: x[3] * np.cos(x[2]),
                                                     x61: x[1] * np.tan(x[0]), x62: x[2] * np.tan(x[0]), x63: x[3] * np.tan(x[0]), x64: x[2] * np.tan(x[1]), x65: x[3] * np.tan(x[1]), x66: x[3] * np.tan(x[2]),
                                                     }))
            dx4_dt = sp.N(ode_list_symbolic[3].subs({x0: 1, x1: x[0], x2: x[1], x3: x[2], x4: x[3],
                                                     x5: x[0] * x[0], x6: x[1] * x[1], x7: x[2] * x[2], x8: x[3] * x[3],
                                                     x9: x[0] * x[1], x10: x[0] * x[2], x11: x[0] * x[3],
                                                     x12: x[1] * x[2], x13: x[1] * x[3], x14: x[2] * x[3],
                                                     x15: x[0] * x[0] * x[1], x16: x[0] * x[0] * x[2], x17: x[0] * x[0] * x[3], x18: x[1] * x[1] * x[2],
                                                     x19: x[1] * x[1] * x[3], x20: x[2] * x[2] * x[3], x21: x[0] * x[0] * x[0],
                                                     x22: x[1] * x[1] * x[1], x23: x[2] * x[2] * x[2], x24: x[3] * x[3] * x[3],
                                                     x25: np.sin(x[0]), x26: np.sin(x[1]), x27: np.sin(x[2]), x28: np.sin(x[3]), x29: np.cos(x[0]), x30: np.cos(x[1]), x31: np.cos(x[2]), x32: np.cos(x[3]), x33: np.tan(x[0]), x34: np.tan(x[1]), x35: np.tan(x[2]), x36: np.tan(x[3]),
                                                     x37: x[0]*np.sin(x[0]), x38: x[1]*np.sin(x[1]), x39: x[2]*np.sin(x[2]), x40: x[3]*np.sin(x[3]), x41: x[0]*np.cos(x[0]), x42: x[1]*np.cos(x[1]), x43: x[2]*np.cos(x[2]), x44: x[3]*np.cos(x[3]), x45: x[0]*np.tan(x[0]), x46: x[1]*np.tan(x[1]), x47: x[2]*np.tan(x[2]), x48: x[3]*np.tan(x[3]),
                                                     x49: x[1]*np.sin(x[0]), x50: x[2]*np.sin(x[0]), x51: x[3]*np.sin(x[0]), x52: x[2]*np.sin(x[1]), x53: x[3]*np.sin(x[1]), x54: x[3]*np.sin(x[2]),
                                                     x55: x[1] * np.cos(x[0]), x56: x[2] * np.cos(x[0]), x57: x[3] * np.cos(x[0]), x58: x[2] * np.cos(x[1]), x59: x[3] * np.cos(x[1]), x60: x[3] * np.cos(x[2]),
                                                     x61: x[1] * np.tan(x[0]), x62: x[2] * np.tan(x[0]), x63: x[3] * np.tan(x[0]), x64: x[2] * np.tan(x[1]), x65: x[3] * np.tan(x[1]), x66: x[3] * np.tan(x[2]),
                                                     }))
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
    if plot_results:
        # Plotting the Result
        fig, ax = plt.subplots()
        ax.plot(sol.t,ca,sol.t,cb,sol.t,cc,sol.t,cd)
        ax.set_xlabel('zeit in sekunden und so')
        ax.set_ylabel('Concentration c in mol/m^3')
        ax.legend(['H2','CO2','CH4','H2O'])
        plt.title('automatic plot of sindy result:')
        plt.show()

    c = [ca, cb, cc, cd]
    return c

