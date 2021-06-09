# test

# import packages/modules
import numpy as np
from data import *
from core import constants as CONST
from rmt import rmtExe
from core.utilities import roundNum
from docs.rmtUtility import rmtUtilityClass as rmtUtil


# operating conditions
# pressure [Pa]
P = 3.5*1e6
# temperature [K]
T = 523

# set feed mole fraction
# H2/COx ratio
H2COxRatio = 2.0
# CO2/CO ratio
CO2COxRatio = 0.8
feedMoFr = setFeedMoleFraction(H2COxRatio, CO2COxRatio)
# print(f"feed mole fraction: {feedMoFr}")

# mole fraction
y0 = np.array([feedMoFr[0], feedMoFr[1], feedMoFr[2],
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
#  molar flowrate @ ideal gas[kmol/s]
Ft0 = rmtUtil.VoFlRaSTPToMoFl(VoFlRaSTP)/1000
#  initial concentration[kmol/m3]
Ct0 = Ft0/VoFlRa


# reactions
reactionSet = {
    "R1": "CO2 + 3H2 <=> CH3OH + H2O",
    "R2": "CO + H2O <=> H2 + CO2",
    "R3": "2CH3OH <=> DME + H2O",
    "RT": "3CO + 3H2 <=> CH3OCH3 + CO2"
}

# model input - feed
modelInput = {
    "model": "M1",
    "operating-conditions": {
        "pressure": P,
        "temperature": T,
    },
    "feed": {
        "mole-fraction": feedMoFr,
        "molar-flowrate": Ft0,
        "molar-flux": Fl0,
        "components": {
            "shell": ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"],
            "tube": [],
            "medium": []
        }
    },
    "reactions": reactionSet
}

# run exe
res = rmtExe(modelInput)
print(f"modeling result: {res}")
