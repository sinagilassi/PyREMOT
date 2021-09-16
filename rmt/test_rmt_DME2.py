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
opT = 50

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
#  flowrate @ P & T [m^3/s]
VoFlRa = InGaVe*rea_CSA
#  flowrate at STP [m^3/s]
VoFlRaSTP = rmtUtil.volumetricFlowrateSTP(VoFlRa, P, T)
#  molar flowrate @ ideal gas [mol/s]
MoFlRa0 = rmtUtil.VoFlRaSTPToMoFl(VoFlRaSTP)
#  initial concentration[mol/m3]
Ct0 = MoFlRa0/VoFlRa
# molar flux
MoFl0 = MoFlRa0/(rea_CSA)

# component all
compList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]

# reactions
reactionSet = {
    "R1": "CO2 + 3H2 <=> CH3OH + H2O",
    "R2": "CO + H2O <=> H2 + CO2",
    "R3": "2CH3OH <=> DME + H2O",
}

reactionRateSet = {
    "R1": "T+ P + y + 1",
    "R2": "T+ P + y + 2",
    "R3": "T+ P + y + 3",
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

# NOTE
# external heat
# overall heat transfer coefficient [J/m^2.s.K]
U = 50
# effective heat transfer area per unit of reactor volume [m^2/m^3]
a = 4/ReInDi
# medium temperature [K]
Tm = 523
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
    "model": "M1",
    "operating-conditions": {
        "pressure": P,
        "temperature": T,
        "period": opT
    },
    "feed": {
        "mole-fraction": MoFri0,
        "molar-flowrate": MoFlRa0,
        "molar-flux": 0,
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
        "CaSpHeCa": CaSpHeCa
    }
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
ssModelingData = res['resModel']['dataYs']

# save modeling result [txt]
# np.savetxt('ssModeling.txt', ssModelingData, fmt='%.10e')
# load
# c = np.loadtxt('ssModeling.txt', dtype=np.float64)
# print("c: ", c, " c Shape: ", c.shape)

# save binary file
np.save('ResM1.npy', ssModelingData)
# load
# b2Load = np.load('res3.npy')
# print("b2Load: ", b2Load, b2Load.shape)
