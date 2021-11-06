# MODEL INPUT VARIABLES
# -----------------------

# import package/module
import numpy as np


# # universal gas constant [J/mol.K]
# R_const = 8.314
# # packed reactor diameter [m]
# # rea_D = 0.0254;
# rea_D = 0.0381
# rea_Di = 0.04579
# # reactor diameter [m]
# reaW_D = 0.0025
# # reactor wall thickness [m]
# reaWall_D = 0.01
# # membrane thickness [m]
# mem_t = 0.0001
# # membrane tube diameter [m2]
# mem_D = 0.0254
# # bed height [m]
# rea_L = 1
# # bed porosity
# bed_por = 0.39
# # catalyst particle density [kg/m^3]
# cat_rho = 1982
# # catalyst porosity
# cat_por = 0.45
# # catalyst tortuosity
# cat_tor = 2
# # voidage of the fixed bed
# rea_por = 0.39
# # catalyst particle diameter [m]
# cat_d = 0.002
# # reactor temperature [K]
# rea_T = 523
# # permeate temperature [K]
# per_T = 523-1
# # reactor pressure [MPa]
# rea_P = 3.5
# # permeate pressure [MPa]
# per_P = 0.35
# # pressure ratio
# per_ratio = rea_P/rea_P  # 0.05
# # medium temperature [K]
# Tm = rea_T-1
# # steam temperature [K]
# St_T = Tm - 1
# # shell pressure [Pa]
# St_P = 101325
# # ambient pressure [Pa]
# amb_P = rea_P
# # inlet fixed bed superficial gas velocity [m/s]
# sgv0 = 0.2
# # inlet permeate side superficial velocity [m/s]
# sgvp0 = 0.5
# # inlet fixed bed interstitial gas velocity [m/s]
# igv0 = sgv0/rea_por
# # inlet permeate interstitial gas velocity [m/s]
# igvp0 = sgvp0/rea_por
# # membrane area [m^2/m^3 reactor]
# mem_A = 100  # 250
# # temperature [K]
# T = rea_T
# # fraction of solids
# rea_solid = 1-rea_por
# # bulk density [kg/m3]
# bulk_rho = cat_rho*rea_solid
# # initial concentration [mol/m^3]
# Cmin = 0.001
# # initial flowrate [mol/s]
# Fmin = 0.001
# # H2O permeance [kmol/(s*m^2*Pa)]
# QH2O = 5e-10
# # H2O/H2 selectivity
# SelH2OH2 = 30  # 10
# # H2 permeance [kmol/(s*m^2*Pa)]
# QH2 = QH2O/SelH2OH2
# # STP condition
# # pressure [Pa]
# Pstp = 101325
# # temperature [K]
# Tstp = 273.15

# # viscosity [kg/m.s]
# miu_g = 1e-5
# # overall heat transfer coefficient [J/s.m2.K]
# U = 100  # 500
# # heat transfer area over volume [m2/m3]
# a = 4/rea_D
# # average Ua [kW/m3.K]
# Ua = U*a/1000
# # bed specific area [m2/m3 solid]
# av = 352
# # reactor wall thermal conductivity [J/K.m.s]
# # iron = 79;
# # steel = 50;
# kwall = 50
# # reference temperature [K]
# Tref = 25 + 273.15
# # thermal conductivity of catalyst [J/K.m.s]
# therCop = 1
# # membrane thermal conductivity [J/K.m.s]
# kmem = 1
# # steam molar flux [kmol/m2.s]
# Fst = 1


# # CORRECTION FACTOR
# # -------------------
# hfsa = 1  # 2.5
# kgi = 1  # 2.5
