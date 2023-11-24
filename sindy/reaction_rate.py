#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:42:30 2023

@author: peterson
"""
import jax.numpy as np
import jax
from Parameter_PDE_CH4 import data



def reaction_rate(T, x, p):
    # Reaction Kinetics data
    # Reference Temperature in Kelvin
    T_ref = 555
    # Pre-exponential factor in mol Pa^-1 gcat^-1 s^-1
    k0 = 0.00000346
    # Activation Energy in J mol^-1
    energy_act = 77.5E3
    # Eq. Konstant Hydrogen in Pa^-0.5
    eq_const_H2_0 = 0.00139140217047409
    # Eq. Constant Hydroxyl in Pa^-0.5
    eq_const_OH_0 = 0.00158113883008419
    # Eq. Constant Mix in Pa^-0.5
    eq_const_mix_0 = 0.00278280434094817
    # Enthalpy Difference Hydrogen in J mol^-1
    dH_H2 = -6.2E3
    # Enthalpy Difference Hydroxyl in J mol^-1
    dH_OH = 22.4E3
    # Enthalpy Difference Mix in J mol^-1
    dH_mix = -10E3

    """Estimate reaction rate according to Koschany et al.."""
    # reaction rate
    temp_dependence = (1/T_ref-1/T)
    # Approximated by an empirical formula (Koschany2016 Eq. 30) [Pa^-2]
    Keq = 137*T**(-3.998)*np.exp(158.7E3/data["R"]/T)
    # Equilibrium Constant in Pa
    Keq = Keq*(1.01325*1e5)**-2

    # Arrhenius dependence
    k = k0 * np.exp(energy_act/data["R"]*temp_dependence)
    # van't Hoff Equations
    K_H2 = eq_const_H2_0*np.exp(dH_H2/data["R"]*temp_dependence)
    K_OH = eq_const_OH_0*np.exp(dH_OH/data["R"]*temp_dependence)
    K_mix = eq_const_mix_0*np.exp(dH_mix/data["R"]*temp_dependence)

    # partial pressures [Pa]
    p_i = x*p
    # unpack partial pressures
    p_H2 = p_i[0, :]
    p_CO2 = p_i[1, :]
    p_CH4 = p_i[2, :]
    p_H2O = p_i[3, :]

    # equation
    r_NUM = k*p_H2**0.5*p_CO2**0.5*(1-(p_CH4*p_H2O**2)/(p_CO2*p_H2**4*Keq))
    r_DEN = (1+K_OH*p_H2O/(p_H2**0.5)+K_H2*p_H2**0.5+K_mix*p_CO2**0.5)
    # mass based reaction rates in mol/(s*kg_cat)
    r = r_NUM/(r_DEN**2)
    return r
