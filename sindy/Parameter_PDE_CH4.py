"""
Solving Partial-Differential Equation 1D-PFTR for Methanantion
Author: Luisa Peterson
"""

import numpy as np

# Define the data dictionary with default values
data = {
    "T_w": 250 + 273.15,  # Reactor wall temperature [K]
    "F_in_Nl_min": 6,  # Norm volume in Nl/min
    "t_end": 300,  # Final time point for the simulation in s
    "Nt": 300,  # Number of time steps
    "Nz": 300,  # Number of control volumes for Finite-Volume-Method

    # Other parameters
    "R": 8.3145,  # Universal gas constant in J/(mol*K)
    "species": ['H2', 'CO2', 'CH4', 'H2O'],  # List with species
    "nue": np.array([-4, -1, 1, 2]),  # Stoichiometric matrix
    # Molar Mass in kg/mol
    "Molar_Mass": np.array([2.01588, 44.0095, 16.0425, 18.01528]) * 1e-3,

    # Reactor Parameters
    "epsilon": 0.39,  # Specific katalyst volume in (m_cat)^3/(m_ges^3)
    "D_R": 0.01,  # Inner diameter reactor in m
    "L_R": 2,  # Reactor length in m
    "V_r": None,  # Reaction volume in m^3
    "p_R": 2E5,  # Reactor pressure in Pa
    "alpha_wall_ext": 100,  # Heat transfer coefficient W/m^2/K [100, 2000]
    "fac_eff": 0.1,  # effectivness factor

    # Inlet Parameters
    "T_gas_in": 270+273.15,  # Gas temperature in K
    "X_0": 1e-5,  # Conversion at time 0 [-]
    "X_in": 1e-5,  # Conversion at inlet [-]
    "n_in_total": None,  # Total molar fraction in mol
    "x_in": np.array([0.8, 0.2, 0, 0]),  # Inlet molar gas composition with dilution[-]
    "n_in": None,  # Inlet molar gas composition in mol
    "w_in": None,  # Inlet molar gas composition by weight
    "M_gas_in": None,  # Molar Mass of the gas
    "rho_fluid_in": None,  # Gas density by ideal gas law in kg/m^3
    "L_to_m3": 1e-3,  # Conversion factor
    "min_to_s": 60.0,  # Conversion factor
    "F_in": None,  # Gas flow rate in m^3/s
    "v_gas_in": None,  # Gas velocity in m/s

    # Catalyst Parameters (taken from Bremer20020_diss page 57)
    "rho_cat": 1032,  # 2355.2,  # Catalyst density in kg/m^3 page 141
    "cp_cat": 1507,  # Heat capacity in J/kg/K
    "dp_cat": 0.0025,  # Catalyst particle diameter in m
    # Parameters for Finite-Volume-Method
    "zeta_mid": None,
    "D_zeta": None,
    "d_zeta": None,
    "zeta": None,
    "t_span": None,
    "t_points": None
}


# Define a function to update dependent values based on current data
def update_dependent_values():
    data["V_r"] = data["L_R"] / 4 * np.pi * data["D_R"] ** 2
    data["n_in_total"] = (data["p_R"] * data["V_r"]
                          / data["R"] / data["T_gas_in"])  # ideal gas law
    data["n_in"] = data["n_in_total"] * data["x_in"]
    data["w_in"] = (data["x_in"] * data["Molar_Mass"]
                    / np.sum(data["x_in"] * data["Molar_Mass"]))
    data["M_gas_in"] = np.sum(data["x_in"] * data["Molar_Mass"])
    data["rho_fluid_in"] = (data["p_R"] * data["M_gas_in"]
                            / data["R"] / data["T_gas_in"])
    data["F_in"] = (data["F_in_Nl_min"] * data["L_to_m3"] / data["min_to_s"]
                    * (1.013E5 / data["p_R"]) * (data["T_gas_in"] / 273.15))
    data["v_gas_in"] = (4 * data["F_in"]) / (np.pi * data["D_R"] ** 2)
    data["zeta_mid"] = np.linspace(0, 1, 2 * data["Nz"] + 1)
    data["D_zeta"] = data["zeta_mid"][2:][::2] - data["zeta_mid"][0: -1][::2]
    data["d_zeta"] = data["zeta_mid"][3:][::2] - data["zeta_mid"][1: -2][::2]
    data["zeta"] = data["zeta_mid"][1:][::2]
    data["t_span"] = np.array([0, data["t_end"]])
    data["t_points"] = np.linspace(0, data["t_end"],
                                   num=data["Nt"])


# Call the update_dependent_values function to compute initial dependent values
update_dependent_values()
