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


# operating conditions
# pressure [Pa]
P = 3*1e5
# temperature [K]
T = 973
# operation period [s]
# [h]
opT = 1

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

# total concentration [kmol/m3]
ct0T = calTotalConcentration(ct0)

# inlet fixed bed superficial gas velocity [m/s]
SuGaVe = 0.2
# inlet fixed bed interstitial gas velocity [m/s]
InGaVe = SuGaVe/bed_por
# flux [kmol/m2.s] -> total concentration x superficial velocity
Fl0 = ct0T*SuGaVe

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
# reaction rates
# initial values
varis0 = {
    # loopVars
    # T,P,NoFri,SpCoi
    # other vars
    # [m^3/(mol*s)]
    "k0": 0.0072,
    "y_CH4": lambda x: x['MoFri'][0],
    "C_CH4": lambda x: x['SpCoi'][0]
}

# reaction rates
rates0 = {
    # [mol/m^3.s]
    "r1": lambda x: x['k0']*(x['C_CH4']**2)
}

# reaction rate
reactionRateSet = {
    "VARS": varis0,
    "RATES": rates0
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

# NOTE
# external heat
# overall heat transfer coefficient [J/m^2.s.K]
U = 50
# effective heat transfer area per unit of reactor volume [m^2/m^3]
a = 4/ReInDi
# medium temperature [K]
Tm = T
# Ua
Ua = U*a
#
externalHeat = {
    "OvHeTrCo": U,
    "EfHeTrAr": a,
    "MeTe": Tm
}

# gas mixture viscosity [Pa.s]
GaMiVi = 1e-5

# model input - feed
modelInput = {
    "model": "M10",
    "operating-conditions": {
        "pressure": P,
        "temperature": T,
        "period": opT
    },
    "feed": {
        "mole-fraction": MoFri0,
        "molar-flowrate": MoFlRa0,
        "molar-flux": 0,
        "superficial-velocity": SuGaVe,
        "volumetric-flowrate": VoFlRa,
        "concentration": ct0,
        "mixture-viscosity": GaMiVi,
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
}

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
