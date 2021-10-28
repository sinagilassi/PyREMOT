# TEST
# STATIC MODELING
# ----------------

# REVIEW
# check unit
# flowrate [mol/s]
# rate formation [mol/m^3.s]

# import packages/modules
import numpy as np
import math
import json
from data import *
from core import constants as CONST
from rmt import rmtExe
from core.utilities import roundNum
from docs.rmtUtility import rmtUtilityClass as rmtUtil


# operating conditions
# pressure [Pa]
P = 5*1e6
# temperature [K]
T = 523
# operation period [s]
opT = 0.5

# set feed mole fraction
# H2/COx ratio
H2COxRatio = 1
# CO2/CO ratio
CO2COxRatio = 0.5
feedMoFr = setFeedMoleFraction(H2COxRatio, CO2COxRatio)
# print(f"feed mole fraction: {feedMoFr}")

# mole fraction
MoFri0 = np.array([feedMoFr[0], feedMoFr[1], feedMoFr[2],
                   feedMoFr[3], feedMoFr[4], feedMoFr[5]])
# print(f"component mole fraction: {y0}")

# concentration [kmol/m3]
ct0 = calConcentration(feedMoFr, P, T)
# print(f"component concentration: {ct0}")

# total concentration [kmol/m3]
ct0T = calTotalConcentration(ct0)
# print(f"total concentration: {ct0T}")

# inlet fixed bed superficial gas velocity [m/s]
SuGaVe = 0.2
# inlet fixed bed interstitial gas velocity [m/s]
InGaVe = SuGaVe/bed_por
# flux [kmol/m2.s] -> total concentration x superficial velocity
Fl0 = ct0T*SuGaVe
# print(f"feed flux: {Ft0}")

#  cross section of reactor x porosity [m2]
rea_CSA = rmtUtil.reactorCrossSectionArea(bed_por, rea_D)
#  flowrate @ P & T [m3/s]
VoFlRa = InGaVe*rea_CSA
#  flowrate at STP [m3/s]
VoFlRaSTP = rmtUtil.volumetricFlowrateSTP(VoFlRa, P, T)
#  molar flowrate @ ideal gas [mol/s]
MoFlRa0 = rmtUtil.VoFlRaSTPToMoFl(VoFlRaSTP)
#  initial concentration[mol/m3]
Ct0 = MoFlRa0/VoFlRa
# initial density [kg/m^3]
# GaDe = Ct0

# component all
compList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]

# reactions
reactionSet = {
    "R1": "CO2 + 3H2 <=> CH3OH + H2O",
    "R2": "CO + H2O <=> H2 + CO2",
    "R3": "2CH3OH <=> DME + H2O",
}

# NOTE
# reactor
# reactor volume [m^3]
ReVo = 5
# reactor length [m]
ReLe = rea_L
# reactor inner diameter [m]
# ReInDi = math.sqrt(ReVo/(ReLe*CONST.PI_CONST))
ReInDi = rea_D
# particle dimeter [m]
PaDi = cat_d
# particle density [kg/m^3]
CaDe = cat_rho
# particle specific heat capacity [kJ/kg.K]
CaSpHeCa = cat_Cp/1000
# catalyst porosity
CaPo = cat_por
# catalyst tortuosity
CaTo = cat_tor
# catalyst thermal conductivity [J/K.m.s]
CaThCo = therCop
# catalyst bed dencity  [kg/m^3]
CaBeDe = bulk_rho

# NOTE
# external heat
# overall heat transfer coefficient [J/m^2.s.K]
U = 50
# effective heat transfer area per unit of reactor volume [m^2/m^3]
a = 4/ReInDi
# medium temperature [K]
Tm = 0
# Ua
Ua = U*a
#
externalHeat = {
    "OvHeTrCo": U,
    "EfHeTrAr": a,
    "MeTe": Tm
}

# gas viscosity [Pa.s]
GaVii = np.array([1, 1, 1, 1, 1, 1])
# gas mixture viscosity [Pa.s]
GaMiVi = 1e-5
# diffusivity coefficient - gas phase [m^2/s]
# GaDii = np.zeros(compNo)  # gas_diffusivity_binary(yi,T,P0);
GaDii = np.array([6.61512999110972e-06,	2.12995183554984e-06,	1.39108654241678e-06,
                  2.20809430865725e-06,	9.64429037148681e-07,	8.74374373632434e-07])
# thermal conductivity - gas phase [J/s.m.K]
# GaThCoi = np.zeros(compNo)  # f(T);
GaThCoi = np.array([0.278863993072407, 0.0353728593093126,	0.0378701882504170,
                    0.0397024608654616,	0.0412093811132403, 0.0457183034548015])
# mixture thermal conductivity - gas phase [J/s.m.K]
# convert
GaThCoMix = 0.125

# NOTE
### TEST ###
# bulk concentration
GaSpCoi = ct0
# mass transfer coefficient [m/s]
MaTrCo0 = np.array([0.0273301866548795,	0.0149179341780856,	0.0108707796723462,
                    0.0157945517381349,	0.0104869502041277,	0.00898673624257253])
# heat transfer coefficient - gas/solid [J/m^2.s.K]
HeTrCo0 = 1731


# NOTE
# reaction rates
# initial values
varis0 = {
    # loopVars
    # T,P,NoFri
    #  mole fraction
    "CaDe": CaDe,
    # catalyst porosity
    "CaPo": CaPo,
    # vars key/value
    "RT": lambda x: x['R_CONST']*x['T'],
    #  kinetic constant
    # DME production
    #  [kmol/kgcat.s.bar2]
    "K1": lambda x: 35.45*math.exp(-1.7069e4/x['RT']),
    #  [kmol/kgcat.s.bar]
    "K2": lambda x: 7.3976*math.exp(-2.0436e4/x['RT']),
    #  [kmol/kgcat.s.bar]
    "K3": lambda x: 8.2894e4*math.exp(-5.2940e4/x['RT']),
    # adsorption constant [1/bar]
    "KH2": lambda x: 0.249*math.exp(3.4394e4/x['RT']),
    "KCO2": lambda x: 1.02e-7*math.exp(6.74e4/x['RT']),
    "KCO": lambda x: 7.99e-7*math.exp(5.81e4/x['RT']),
    #  equilibrium constant
    "Ln_KP1": lambda x: 4213/x['T'] - 5.752 * \
    math.log(x['T']) - 1.707e-3*x['T'] + 2.682e-6 * \
    (math.pow(x['T'], 2)) - 7.232e-10*(math.pow(x['T'], 3)) + 17.6,
    "KP1": lambda x: math.exp(x['Ln_KP1']),
    "log_KP2": lambda x: 2167/x['T'] - 0.5194 * \
    math.log10(x['T']) + 1.037e-3*x['T'] - 2.331e-7 * \
    (math.pow(x['T'], 2)) - 1.2777,
    "KP2": lambda x: math.pow(10, x['log_KP2']),
    "Ln_KP3": lambda x: 4019/x['T'] + 3.707 * \
    math.log(x['T']) - 2.783e-3*x['T'] + 3.8e-7 * \
    (math.pow(x['T'], 2)) - 6.56e-4/(math.pow(x['T'], 3)) - 26.64,
    "KP3": lambda x: math.exp(x['Ln_KP3']),
    #  mole fraction
    "yi_H2": lambda x: x['MoFri'][0],
    "yi_CO2": lambda x: x['MoFri'][1],
    "yi_H2O": lambda x: x['MoFri'][2],
    "yi_CO": lambda x: x['MoFri'][3],
    "yi_CH3OH": lambda x: x['MoFri'][4],
    "yi_DME": lambda x: x['MoFri'][5],
    # partial pressure
    #  partial pressure of H2 [bar]
    "PH2": lambda x: x['P']*(x['yi_H2'])*1e-5,
    #  partial pressure of CO2 [bar]
    "PCO2": lambda x: x['P']*(x['yi_CO2'])*1e-5,
    #  partial pressure of H2O [bar]
    "PH2O": lambda x: x['P']*(x['yi_H2O'])*1e-5,
    #  partial pressure of CO [bar]
    "PCO": lambda x: x['P']*(x['yi_CO'])*1e-5,
    #  partial pressure of CH3OH [bar]
    "PCH3OH": lambda x: x['P']*(x['yi_CH3OH'])*1e-5,
    #  partial pressure of CH3OCH3 [bar]
    "PCH3OCH3": lambda x: x['P']*(x['yi_DME'])*1e-5,
    # reaction rates
    "ra1": lambda x: x['PCO2']*x['PH2'],
    "ra2": lambda x: 1 + (x['KCO2']*x['PCO2']) + (x['KCO']*x['PCO']) + math.sqrt(x['KH2']*x['PH2']),
    "ra3": lambda x: (1/x['KP1'])*((x['PH2O']*x['PCH3OH'])/(x['PCO2']*(math.pow(x['PH2'], 3)))),
    "ra4": lambda x: x['PH2O'] - (1/x['KP2'])*((x['PCO2']*x['PH2'])/x['PCO']),
    "ra5": lambda x: (math.pow(x['PCH3OH'], 2)/x['PH2O'])-(x['PCH3OCH3']/x['KP3']),
}

# reaction rates
rates0 = {
    "r1": lambda x: x['K1']*(x['ra1']/(math.pow(x['ra2'], 3)))*(1-x['ra3'])*x['CaDe'],
    "r2": lambda x: x['K2']*(1/x['ra2'])*x['ra4']*x['CaDe'],
    "r3": lambda x: x['K3']*x['ra5']*x['CaDe']
}

# reaction rate
reactionRateSet = {
    "VARS": varis0,
    "RATES": rates0
}

# NOTE
# model ids
# M11
# T1, T2
# M12

# model input - feed
modelInput = {
    "model": "M14",
    "operating-conditions": {
        "pressure": P,
        "temperature": T,
        "period": opT,
        "process-type": "non-iso-thermal",
        "numerical-method": "fdm"
    },
    "feed": {
        "mole-fraction": MoFri0,
        "molar-flowrate": MoFlRa0,
        "molar-flux": 0,
        "superficial-velocity": SuGaVe,
        "volumetric-flowrate": VoFlRa,
        "concentration": ct0,
        "viscosity": GaVii,
        "mixture-viscosity": GaMiVi,
        "diffusivity": GaDii,
        "thermal-conductivity": GaThCoi,
        "mixture-thermal-conductivity": GaThCoMix,
        "components": {
            "shell": compList,
            "tube": [],
            "medium": []
        }
    },
    "reactions": reactionSet,
    "reaction-rates": reactionRateSet,
    "external-heat": externalHeat,
    "reactor": {
        "ReInDi": ReInDi,
        "ReLe": ReLe,
        "PaDi": PaDi,
        "BeVoFr": bed_por,
        "CaBeDe": bulk_rho,
        "CaDe": CaDe,
        "CaSpHeCa": CaSpHeCa,
        "CaPo": CaPo,
        "CaTo": CaTo,
        "CaThCo": CaThCo
    },
    "solver-config": {
        "ivp": "LSODA",
        "root": "fsolve",
        "mesh": "normal"
    },
    "test-const": {
        "numerical-method": "fem",
        "Cbi": GaSpCoi,
        "Tb": T,
        "MaTrCo0": MaTrCo0,
        "HeTrCo0": HeTrCo0
    }
}

# NOTE
# fsolve, then adjust initial guess 0.5, fdm
# oc, initial guess would be dimensionless
# root
# fsolve
# least_squares
# Radau
# LSODA
# BDF
# normal

# run exe
res = rmtExe(modelInput)
# print(f"modeling result: {res}")

# save modeling result
# with open('res.json', 'w') as f:
#     json.dump(res, f)

# steady-state results
# concentration
# total concentration
# ssModelingData = res['resModel']['dataYs']

# save modeling result [txt]
# np.savetxt('ssModeling.txt', ssModelingData, fmt='%.10e')
# load
# c = np.loadtxt('ssModeling.txt', dtype=np.float64)
# print("c: ", c, " c Shape: ", c.shape)

# save binary file
# np.save('ssModeling.npy', ssModelingData)
# load
# b2Load = np.load('res3.npy')
# print("b2Load: ", b2Load, b2Load.shape)
