# SAMPLE REACTION LIST
# --------------------

# import package/modules
import math
# internals
from data import *

# NOTE
### constants/initial data ###
# catalyst porosity
CaPo = cat_por
# catalyst tortuosity
CaTo = cat_tor
# particle density [kg/m^3]
CaDe = cat_rho

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
