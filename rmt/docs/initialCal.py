# initialize modeling

# import package/module
import data
from .rmtUtility import *


# # temperature [K]
# T0 = data.T
# # pressure [Pa]
# P0 = data.rea_P*1e6
# # stp conditions
# Tstp = data.Tstp
# Pstp = data.Pstp
# # voidage of packed bed
# por = data.rea_por
# # interstitial velocity [m/s]
# Uig = data.igv0
# # cross section of reactor x porosity [m2]
# rea_CSA = por*(3.14*(data.rea_D**2)/4)
# # flowrate @ P & T [m3/s]
# v0 = Uig*rea_CSA
# # flowrate at STP [m3/s]
# v0stp = v0*(P0/Pstp)*(Tstp/T0)
# # molar flowrate @ ideal gas [kmol/s]
# Ft0 = (v0stp/0.02241)/1000
# # initial concentration [kmol/m3]
# Ct0 = Ft0/v0

# feed mole fraction
# y0_H2 = data.feedMoFri[0]
# y0_CO2 = data.feedMoFri[1]
# y0_H2O = data.feedMoFri[2]
# y0_CO = data.feedMoFri[3]
# y0_CH3OH = data.feedMoFri[4]
# y0_DME = data.feedMoFri[5]

# molar ratio H2/[CO+CO2]
# mr1 = data.y0_H2/(y0_CO2 + y0_CO)

# feed concentration [kmol/m^3]
# ct0_H2 = (P0/(data.R_const*T0))*y0_H2/1000
# ct0_CO2 = (P0/(data.R_const*T0))*y0_CO2/1000
# ct0_H2O = (P0/(data.R_const*T0))*y0_H2O/1000
# ct0_CO = (P0/(data.R_const*T0))*y0_CO/1000
# ct0_CH3OH = (P0/(data.R_const*T0))*y0_CH3OH/1000
# ct0_DME = (P0/(data.R_const*T0))*y0_DME/1000
# ct cal
# Ct0Mix = ct0_H2 + ct0_CO2 + ct0_H2O + ct0_CO + ct0_CH3OH + ct0_DME
# molecular weight [kg/kmol]
# MW_mix = \
#     y0_H2*data.MW_H2 + \
#     y0_CO2*data.MW_CO2 + \
#     y0_H2O*data.MW_H2O + \
#     y0_CO*data.MW_CO + \
#     y0_CH3OH*data.MW_CH3OH + \
#     y0_DME*data.MW_DME

# MW_mix = mixtureMolecularWeight(data.feedMWi, data.feedMoFri)

# density [kg/m3]
# rho = Ct0*MW_mix


# # initial superficial velocity [m/s]
# sgv0 = data.sgv0
# # initial interstitial velocity [m/s]
# igv0 = data.igv0
# # catalyst particle diameter
# cat_d = data.cat_d
# # catalyst surface area
# cat_A = 4*3.14*(cat_d**2)
# # catalyst volume
# cat_V = (4/3)*3.14*(cat_d**3)
# # equivalent particle diameter
# Dp = 6*(cat_V/cat_A)
# # viscosity [kg/m.s]
# miu_g = data.miu_g
