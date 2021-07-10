# PLUG-FLOW REACTOR MODEL
# -------------------------

# import packages/modules
import math as MATH
import numpy as np
from library.plot import plotClass as pltc
from scipy.integrate import solve_ivp
# internal
from core.errors import errGeneralClass as errGeneral
from data.inputDataReactor import *
from core import constants as CONST
from core.utilities import roundNum
from core.config import REACTION_RATE_ACCURACY
from .rmtUtility import rmtUtilityClass as rmtUtil
from .rmtThermo import *


class PlugFlowReactorClass:
    # def main():
    """
    Plug-flow Reactor Model
    assumptions:
        no dispersion, radial gradient in temperature, velocity, concentration, and reaction rate
        species concentration and temperature vary with position along the reactor length
    """
    # internal data
    _internalData = []

    def __init__(self, modelInput, internalData, reactionListSorted):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted

    @property
    def internalData(cls):
        return cls._internalData

    @internalData.setter
    def internalData(cls, value):
        cls._internalData.clear()
        cls._internalData.extend(value)

    def runM10(self):
        """
        M1 modeling case
        """

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']

        # component list
        compList = self.modelInput['feed']['components']['shell']
        labelList = compList.copy()
        labelList.append("Flux")

        # initial values
        # -> mole fraction
        MoFri = self.modelInput['feed']['mole-fraction']
        # -> flux [kmol/m^2.s]
        MoFl = self.modelInput['feed']['molar-flux']
        IV = []
        IV.extend(MoFri)
        IV.append(MoFl)
        # print(f"IV: {IV}")

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, rea_L])
        times = np.linspace(t[0], t[1], 20)
        # tSpan = np.linspace(0, rea_L, 25)

        # ode call
        sol = solve_ivp(PlugFlowReactorClass.modelEquationM1,
                        t, IV, method="LSODA", t_eval=times, args=(P, T))

        # ode result
        successStatus = sol.success
        dataX = sol.t
        dataYs = sol.y

        # check
        if successStatus is True:
            # plot setting
            XYList = pltc.plots2DSetXYList(dataX, dataYs)
            # -> label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # plot result
            pltc.plots2D(dataList, "Reactor Length (m)",
                         "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        else:
            XYList = []
            dataList = []

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def runM1(self):
        """
        M1 modeling case
        """

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")

        # component no
        compNo = len(compList)
        indexTemp = compNo

        # -> mole fraction
        MoFri = self.modelInput['feed']['mole-fraction']

        # external heat
        ExHe = self.modelInput['external-heat']

        # initial values
        IV = []
        IV.extend(MoFri)
        IV.append(T)
        print(f"IV: {IV}")
        IVSet = np.array(IV)

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, rea_L])
        t_span = np.array([0, rea_L])
        times = np.linspace(t_span[0], t_span[1], 100)
        # tSpan = np.linspace(0, rea_L, 25)

        # ode call
        sol = solve_ivp(PlugFlowReactorClass.modelEquationM2,
                        t, IVSet, method="LSODA", t_eval=times, args=(P, compList, StHeRe25, reactionListSorted, ExHe))

        # ode result
        successStatus = sol.success
        dataX = sol.t
        # all results
        dataYs = sol.y
        # concentration
        dataYs1 = sol.y[0:compNo, :]
        labelListYs1 = labelList[0:compNo]
        # temperature
        dataYs3 = sol.y[indexTemp, :]
        labelListYs3 = labelList[indexTemp]

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataX, dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo], dataList[indexTemp]]
            # subplot result
            pltc.plots2DSub(dataLists, "Reactor Length (m)",
                            "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

            # plot result
            # pltc.plots2D(dataList[0:compNo], "Reactor Length (m)",
            #              "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexFlux], "Reactor Length (m)",
            #              "Flux (kmol/m^2.s)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexTemp], "Reactor Length (m)",
            #              "Temperature (K)", "1D Plug-Flow Reactor")

        else:
            XYList = []
            dataList = []

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM1(t, y, P, T):
        """
            M1 model
            mass balance equations
            modelParameters:
                pressure [Pa]
                temperature [K]
        """
        # operating conditions
        # P = modelParameters['pressure']
        # T = modelParameters['temperature']
        # # components
        # comp = modelParameters['components']
        # # components no
        # compNo = len(comp)

        #! loop vars
        # MoFri = np.copy(y)
        yi_H2 = y[0]
        yi_CO2 = y[1]
        yi_H2O = y[2]
        yi_CO = y[3]
        yi_CH3OH = y[4]
        yi_DME = y[5]

        # molar flux [kmol/m^2.s]
        MoFl = y[6]

        # mole fraction list
        MoFri = [yi_H2, yi_CO2, yi_H2O, yi_CO, yi_CH3OH, yi_DME]

        # kinetics
        Ri = PlugFlowReactorClass.modelReactions(P, T, MoFri)
        #  H2
        R_H2 = -(3*Ri[0]-Ri[1])
        # CO2
        R_CO2 = -(Ri[0]-Ri[1])
        # H2O
        R_H2O = (Ri[0]-Ri[1]+Ri[2])
        # CO
        R_CO = -(Ri[1])
        # CH3OH
        R_CH3OH = -(2*Ri[2]-Ri[0])
        # DME
        R_DME = (Ri[2])
        # total
        R_T = -(2*Ri[0])

        # mass balance equation
        # loop vars
        A1 = 1/MoFl
        B1 = 1

        #  H2
        dxdt_H2 = A1*(R_H2 - y[0]*R_T)
        #  CO2
        dxdt_CO2 = A1*(R_CO2 - y[1]*R_T)
        #  H2O
        dxdt_H2O = A1*(R_H2O - y[2]*R_T)
        #  CO
        dxdt_CO = A1*(R_CO - y[3]*R_T)
        #  CH3OH
        dxdt_CH3OH = A1*(R_CH3OH - y[4]*R_T)
        #  DME
        dxdt_DME = A1*(R_DME - y[5]*R_T)
        #  overall
        dxdt_T = B1*R_T
        # build diff/dt
        dxdt = [dxdt_H2, dxdt_CO2, dxdt_H2O,
                dxdt_CO, dxdt_CH3OH, dxdt_DME, dxdt_T]
        return dxdt

    def modelReactions(P, T, y):
        try:
            # pressure [Pa]
            # temperature [K]
            # print("y", y)
            # parameters
            RT = CONST.R_CONST*T

            #  kinetic constant
            # DME production
            #  [kmol/kgcat.s.bar2]
            K1 = 35.45*MATH.exp(-1.7069e4/RT)
            #  [kmol/kgcat.s.bar]
            K2 = 7.3976*MATH.exp(-2.0436e4/RT)
            #  [kmol/kgcat.s.bar]
            K3 = 8.2894e4*MATH.exp(-5.2940e4/RT)
            # adsorption constant [1/bar]
            KH2 = 0.249*MATH.exp(3.4394e4/RT)
            KCO2 = 1.02e-7*MATH.exp(6.74e4/RT)
            KCO = 7.99e-7*MATH.exp(5.81e4/RT)
            #  equilibrium constant
            Ln_KP1 = 4213/T - 5.752 * \
                MATH.log(T) - 1.707e-3*T + 2.682e-6 * \
                (MATH.pow(T, 2)) - 7.232e-10*(MATH.pow(T, 3)) + 17.6
            KP1 = MATH.exp(Ln_KP1)
            log_KP2 = 2167/T - 0.5194 * \
                MATH.log10(T) + 1.037e-3*T - 2.331e-7*(MATH.pow(T, 2)) - 1.2777
            KP2 = MATH.pow(10, log_KP2)
            Ln_KP3 = 4019/T + 3.707 * \
                MATH.log(T) - 2.783e-3*T + 3.8e-7 * \
                (MATH.pow(T, 2)) - 6.56e-4/(MATH.pow(T, 3)) - 26.64
            KP3 = MATH.exp(Ln_KP3)
            #  total concentration
            #  Ct = y(1) + y(2) + y(3) + y(4) + y(5) + y(6);
            #  mole fraction
            yi_H2 = y[0]
            yi_CO2 = y[1]
            yi_H2O = y[2]
            yi_CO = y[3]
            yi_CH3OH = y[4]
            yi_DME = y[5]

            #  partial pressure of H2 [bar]
            PH2 = P*(yi_H2)*1e-5
            #  partial pressure of CO2 [bar]
            PCO2 = P*(yi_CO2)*1e-5
            #  partial pressure of H2O [bar]
            PH2O = P*(yi_H2O)*1e-5
            #  partial pressure of CO [bar]
            PCO = P*(yi_CO)*1e-5
            #  partial pressure of CH3OH [bar]
            PCH3OH = P*(yi_CH3OH)*1e-5
            #  partial pressure of CH3OCH3 [bar]
            PCH3OCH3 = P*(yi_DME)*1e-5

            #  reaction rate expression [kmol/m3.s]
            ra1 = PCO2*PH2
            ra2 = 1 + (KCO2*PCO2) + (KCO*PCO) + MATH.sqrt(KH2*PH2)
            ra3 = (1/KP1)*((PH2O*PCH3OH)/(PCO2*(MATH.pow(PH2, 3))))
            r1 = K1*(ra1/(MATH.pow(ra2, 3)))*(1-ra3)*bulk_rho
            ra4 = PH2O - (1/KP2)*((PCO2*PH2)/PCO)
            r2 = K2*(1/ra2)*ra4*bulk_rho
            ra5 = (MATH.pow(PCH3OH, 2)/PH2O)-(PCH3OCH3/KP3)
            r3 = K3*ra5*bulk_rho

            # result
            # r = roundNum([r1, r2, r3], REACTION_RATE_ACCURACY)
            r = [r1, r2, r3]
            return r
        except Exception as e:
            print(e)
            raise

    def modelEquationM2(t, y, P, comList, StHeRe25, reactionListSorted, ExHe):
        """
            M2 model
            mass and energy balance equations
            modelParameters:
                pressure [Pa]
                compList: component list
                StHeRe25: standard heat of reaction at 25C
                reactionListSorted: reaction list
        """
        # REVIEW
        # t
        # print(f"t: {t}")
        # components no
        # y: component mole fraction, molar flux, temperature
        compNo = len(comList)
        indexT = compNo

        # mole fraction list
        MoFlRai = y[0:compNo]

        # temperature [K]
        T = y[indexT]

        # volumetric flowrate [m^3/s]
        VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

        # concentration species [mol/m^3]
        CoSpi = calConcentrationIG(MoFlRai, VoFlRai)

        # mole fraction
        MoFri = rmtUtil.moleFractionFromConcentrationSpecies(CoSpi)

        # kinetics
        # Ri = np.array(PlugFlowReactorClass.modelReactions(P, T, MoFri))
        # forward frequency factor
        A1 = 8.2e14
        # forward activation energy [J/mol]
        E1 = 284.5e3
        # rate constant [1/s]
        k1 = A1*np.exp(-E1/(R_CONST*T))
        # net reaction rate expression [mol/m^3.s]
        Ri = k1*CoSpi[0]
        # component formation rate [mol/m^3.s]

        # enthalpy
        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K]
        # Cp mean list
        CpMeanList = calMeanHeatCapacityAtConstantPressure(comList, T)
        # print(f"Cp mean list: {CpMeanList}")
        # Cp mixture
        CpMeanMixture = calMixtureHeatCapacityAtConstantPressure(
            MoFri, CpMeanList)
        # print(f"Cp mean mixture: {CpMeanMixture}")

        # enthalpy change from Tref to T [kJ/kmol]
        # enthalpy change
        EnChList = np.array(calEnthalpyChangeOfReaction(reactionListSorted, T))
        # heat of reaction at T [kJ/kmol]
        HeReT = np.array(EnChList + StHeRe25)
        # overall heat of reaction
        OvHeReT = np.dot(Ri, HeReT)

        # external heat

        # FIXME
        # cooling temperature [K]
        Tm = ExHe['MeTe']
        # overall heat transfer coefficient [J/s.m2.K]
        U = ExHe['OvHeTrCo']
        #  heat transfer area over volume [m2/m3]
        a = ExHe['EfHeTrAr']
        # external heat
        Qm = U*a*(Tm - T)

        # diff/dt
        dxdt = []
        # loop vars
        A1 = 1/1
        B1 = 1
        C1 = 1/(1*CpMeanMixture)

        # mass balance (molar flowrate) [mol/s]
        for i in range(compNo):
            dxdt_F = Ri
            dxdt.append(dxdt_F)

        # energy balance (temperature) [K]
        dxdt_T = C1*(-OvHeReT + Qm)

        return dxdt
