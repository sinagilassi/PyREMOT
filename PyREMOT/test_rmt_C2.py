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

# NOTE
### operating conditions ###
# pressure [Pa]
P = 3*1e5
# temperature [K]
T = 973
# operation period [s]
opT = 10

# NOTE
### reactions ###
# component all
compList = ["CH4", "C2H4", "H2"]

# reactions
# ignore: "R2": "CH4 <=> C + 2H2",
reactionSet = {
    "R1": "2CH4 <=> C2H4 + 2H2",
}

# set feed mole fraction
MoFr_H2 = 0.05
MoFr_C2H4 = 0.05
MoFr_CH4 = 1 - (MoFr_H2 + MoFr_C2H4)

# inlet fixed bed superficial gas velocity [m/s]
SuGaVe = 0.01

# NOTE
### reactor ###
# voidage of the fixed bed
rea_por = 0.39
# solid fraction
rea_sol = 1-rea_por
# catalyst particle density (per catalyst volume) [kg/m^3]
cat_rho = 1982
# bulk density (per reactor volume) [kg/m^3]
bulk_rho = cat_rho*rea_sol
# fraction of solids
rea_solid = 1-rea_por
# reactor diameter [m]
rea_dia = 0.007
# reactor radius [m]
rea_rad = rea_dia/2
# reactor length [m]
rea_len = 1  # 0.011
# reactor cross sectional area [m^2]
rea_Ac = CONST.PI_CONST*(rea_rad**2)
# reactor volume [m^3]
rea_vol = (CONST.PI_CONST*(rea_rad**2)*rea_len)
# bulk density [kg/m^3]
bulk_rho0 = bulk_rho
bulk_rho1 = 260
# catalyst mass [kg]
cat_m = bulk_rho1*rea_vol
# reactor volume (real) [m^3]
rea_vol0 = rea_vol*rea_por
# catalyst heat capacity at constant pressure [J/kg·K]
cat_cp = 960
# catalyst thermal conductivity [J/s.m.K]
cat_ThCo = 0.22
# catalyst bed volume [m^3]
catBed_Vol = rea_vol*rea_solid

# NOTE
# reactor
# reactor volume [m^3]
ReVo = 5
# reactor length [m]
ReLe = rea_len
# reactor inner diameter [m]
# ReInDi = math.sqrt(ReVo/(ReLe*CONST.PI_CONST))
ReInDi = rea_dia
# particle dimeter [m]
PaDi = cat_d
# particle density [kg/m^3]
CaDe = cat_rho
# particle specific heat capacity [kJ/kg.K]
CaSpHeCa = cat_cp/1000
# catalyst bed dencity  [kg/m^3]
CaBeDe = bulk_rho

# NOTE
### calculate ###
# mole fraction
MoFri0 = np.array([MoFr_CH4, MoFr_C2H4, MoFr_H2])
# concentration [kmol/m3]
ct0 = calConcentration(MoFri0, P, T, 'kmol/m^3')
# total concentration [kmol/m3]
ct0T = calTotalConcentration(ct0)

# inlet fixed bed interstitial gas velocity [m/s]
InGaVe = SuGaVe/bed_por
# flux [kmol/m2.s] -> total concentration x superficial velocity
Fl0 = ct0T*SuGaVe

# cross section of reactor x porosity [m^2]
rea_CSA = rmtUtil.reactorCrossSectionArea(bed_por, ReInDi)
# real flowrate @ P & T [m^3/s]
VoFlRa = InGaVe*rea_CSA
#  flowrate at STP [m^3/s]
VoFlRaSTP = rmtUtil.volumetricFlowrateSTP(VoFlRa, P, T)
#  molar flowrate @ ideal gas [mol/s]
MoFlRa0 = rmtUtil.VoFlRaSTPToMoFl(VoFlRaSTP)
#  initial concentration[mol/m3]
Ct0 = MoFlRa0/VoFlRa
# molar flux
MoFl0 = MoFlRa0/(rea_CSA/bed_por)
# or
MoFl0_2 = Ct0*InGaVe*bed_por


# NOTE
# external heat
# overall heat transfer coefficient [J/m^2.s.K]
U = 50
# effective heat transfer area per unit of reactor volume [m^2/m^3]
a = 4/ReInDi
# medium temperature [K]
Tm = 0  # T
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

# NOTE
# reaction rates
# initial values
# varis0 = {
#     # loopVars
#     # T,P,NoFri,SpCoi
#     # other vars
#     "bulk_rho1": bulk_rho1,  # [kg/m^3]
#     "krTref": 2.44e-5,  # [variable]
#     "EA": 18.96*1000,  # [J/mol]
#     "KxTref": 0.87,		# [1/bar]
#     "dH": 87.39*1000,  # [J/mol]
#     "Tref":	973.15,  # [K]
#     "RTref": lambda x: x['R_CONST']*x['Tref'],  # [J/mol]
#     "tetaEj": lambda x: x['EA']/x['RTref'],
#     "tetakj": lambda x: math.log(x['krTref']),
#     "kj": lambda x: math.exp(x['tetakj'])*math.exp(x['tetaEj']*(1 - (x['Tref']/x['T']))),
#     "tetaKi": lambda x: math.log(x['KxTref']),
#     "tetaHi": lambda x: x['dH']/x['RTref'],
#     "Ki": lambda x: math.exp(x['tetaKi'])*math.exp(x['tetaHi']*(1 - (x['Tref']/x['T']))),
#     "y_CH4": lambda x: x['MoFri'][0]*x['P']*1e-5,  # [bar]
#     "rA": lambda x: math.sqrt(x['Ki']*x['y_CH4']),
#     "rB": lambda x:	1 + x['rA'],
#     "rC": lambda x:	x['kj']*x['rA']/(x['rB']**2)
# }

# reaction rates
# rates0 = {
#     # [mol/m^3.s]
#     "r1": lambda x: (x['kj']*x['rA']/(x['rB']**2))*x['bulk_rho1']/60
# }

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

# model: M2
# model: N2
# NOTE
# model input - feed
modelInput = {
    "model": "N1",
    "operating-conditions": {
        "pressure": P,
        "temperature": T,
        "period": opT
    },
    "feed": {
        "mole-fraction": MoFri0,
        "molar-flowrate": MoFlRa0,
        "molar-flux": MoFl0,
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
    },
    "solver-config": {
        "ivp": "default"
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
# ssModelingData = res['resModel']['dataYs']

# save modeling result [txt]
# np.savetxt('ssModeling.txt', ssModelingData, fmt='%.10e')
# load
# c = np.loadtxt('ssModeling.txt', dtype=np.float64)
# print("c: ", c, " c Shape: ", c.shape)

# save binary file
# np.save('ResM1.npy', ssModelingData)
# load
# b2Load = np.load('res3.npy')
# print("b2Load: ", b2Load, b2Load.shape)
