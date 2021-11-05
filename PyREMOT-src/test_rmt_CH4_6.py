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
from PyREMOT import rmtExe
from core.utilities import roundNum
from docs.rmtUtility import rmtUtilityClass as rmtUtil


# NOTE
# operating conditions
# pressure [Pa]
P = 3*1e5
# temperature [K]
T = 973
# operation period [s]
# [h]
opT = 0.3

# component all
compList = ["CH4", "C2H4", "H2"]
# reactions
reactionSet = {
    "R1": "2CH4 <=> C2H4 + 2H2",
}

# set feed mole fraction
feedMoFr = [0.9, 0.05, 0.05]

# mole fraction
MoFri0 = np.array([feedMoFr[0], feedMoFr[1], feedMoFr[2]])

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

# NOTE
# gas viscosity [Pa.s]
GaVii = np.array([1, 1, 1])
# gas mixture viscosity [Pa.s]
GaMiVi = 1e-5
# diffusivity coefficient - gas phase [m^2/s]
# GaDii = np.zeros(compNo)  # gas_diffusivity_binary(yi,T,P0);
GaDii = np.array(
    [6.61512999110972e-06,	2.12995183554984e-06, 1.39108654241678e-06])
# thermal conductivity - gas phase [J/s.m.K]
# GaThCoi = np.zeros(compNo)  # f(T);
GaThCoi = np.array([0.278863993072407, 0.0353728593093126, 0.0378701882504170])
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
    # T,P,NoFri,SpCoi
    # other vars
    # [m^3/(kmol*s)]
    "k0": 0.0072*1e3,
    "y_CH4": lambda x: x['MoFri'][0],
    "C_CH4": lambda x: x['SpCoi'][0]
}

# reaction rates
rates0 = {
    # [kmol/m^3.s]
    "r1": lambda x: x['k0']*(x['C_CH4']**2)
}

# reaction rate
reactionRateSet = {
    "VARS": varis0,
    "RATES": rates0
}

# NOTE
# model ids
# M11: hetero finite difference method
# T1: isothermal particle
# T2: non-isothermal particle
# M12: hetero orthogonal collocation method
# M13: hetero model with ode ivp/bvp
# M14: hetero model (reaction occurs at the catalyst surface)

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

# root
# fsolve
# least_squares
# Radau
# LSODA
# BDF
# AM
# refine
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
