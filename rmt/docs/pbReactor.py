# PACKED-BED REACTOR MODEL
# -------------------------

# import packages/modules
import numpy as np
from data.inputDataReactor import *
from scipy.integrate import solve_bvp
from core import constants as CONST


# class PackedBedReactorClass:
def main():
    """
        Packed-bed Reactor Model
    """
    pass


def runM1():
    """
        M1 modeling case
    """

    res = 1
    return res

# model equations


def modelEquationM1(t, y, modelParameters):
    """
        M1 model
        mass balance equations
        modelParameters: 
            pressure [Pa] 
            temperature [K]
    """
    # operating conditions
    P = modelParameters['pressure']
    T = modelParameters['temperature']

    # components
    comp = modelParameters['components']
    # components no
    compNo = len(comp)

    # mole fraction list
    MoFri = np.zeros(compNo)
    MoFri = np.copy(y)

    pass


def modelReactions(P, T, y):
    # pressure [Pa]
    # temperature [K]

    #  kinetic constant
    # DME production
    #  [kmol/kgcat.s.bar2]
    K1 = 35.45*np.exp(-1.7069e4/(CONST.R_CONST*T))*1
    #  [kmol/kgcat.s.bar]
    K2 = 7.3976*np.exp(-2.0436e4/(CONST.R_CONST*T))*1
    #  [kmol/kgcat.s.bar]
    K3 = 8.2894e4*np.exp(-5.2940e4/(CONST.R_CONST*T))*1
    # adsorption constant [1/bar]
    KH2 = 0.249*np.exp(3.4394e4/(CONST.R_CONST*T))
    KCO2 = 1.02e-7*np.exp(6.74e4/(CONST.R_CONST*T))
    KCO = 7.99e-7*np.exp(5.81e4/(CONST.R_CONST*T))
    #  equilibrium constant
    Ln_KP1 = 4213/T - 5.752*np.log(T) - 1.707e-3 * \
        T + 2.682e-6*(T ^ 2) - 7.232e-10*(T ^ 3) + 17.6
    KP1 = np.exp(Ln_KP1)
    log_KP2 = 2167/T - 0.5194 * \
        np.log10(T) + 1.037e-3*T - 2.331e-7*(T ^ 2) - 1.2777
    KP2 = np.power(10, log_KP2)
    Ln_KP3 = 4019/T + 3.707*np.log(T) - 2.783e-3 * \
        T + 3.8e-7*(T ^ 2) - 6.56e-4/(T ^ 3) - 26.64
    KP3 = np.exp(Ln_KP3)
    #  total concentration
    #  Ct = y(1) + y(2) + y(3) + y(4) + y(5) + y(6);
    #  mole fraction
    yi_H2 = y[1]
    yi_CO2 = y[2]
    yi_H2O = y[3]
    yi_CO = y[4]
    yi_CH3OH = y[5]
    yi_DME = y[6]

    #  partial pressure of H2 [bar]
    PH2 = P*(y[1])*1e-5
    #  partial pressure of CO2 [bar]
    PCO2 = P*(y[2])*1e-5
    #  partial pressure of H2O [bar]
    PH2O = P*(y[3])*1e-5
    #  partial pressure of CO [bar]
    PCO = P*(y[4])*1e-5
    #  partial pressure of CH3OH [bar]
    PCH3OH = P*(y[5])*1e-5
    #  partial pressure of CH3OCH3 [bar]
    PCH3OCH3 = P*(y[6])*1e-5

    #  reaction rate expression [kmol/m3.s]
    ra1 = PCO2*PH2
    ra2 = 1+KCO2*PCO2+KCO*PCO+np.sqrt(KH2*PH2)
    ra3 = (1/KP1)*((PH2O*PCH3OH)/(PCO2*(PH2 ^ 3)))
    r1 = K1*(ra1/(ra2 ^ 3))*(1-ra3)*bulk_rho
    ra4 = PH2O-(1/KP2)*((PCO2*PH2)/PCO)
    r2 = K2*(1/ra2)*ra4*bulk_rho
    ra5 = ((PCH3OH ^ 2)/PH2O)-(PCH3OCH3/KP3)
    r3 = K3*ra5*bulk_rho

    # result
    r = [r1, r2, r3]
    return r
