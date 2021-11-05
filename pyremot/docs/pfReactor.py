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
from .rmtReaction import reactionRateExe, componentFormationRate
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
    class initialization:
        model input
        internal data (thermodynamic database)
        reaction list sorted (reactant/product of each reaction)
        reaction stoichiometric coefficient list
    """
    # internal data
    _internalData = []

    def __init__(self, modelInput, internalData, reactionListSorted, reactionStochCoeffList):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted
        self.reactionStochCoeffList = reactionStochCoeffList

    @property
    def internalData(cls):
        return cls._internalData

    @internalData.setter
    def internalData(cls, value):
        cls._internalData.clear()
        cls._internalData.extend(value)

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

        # mole fraction
        MoFri = np.array(self.modelInput['feed']['mole-fraction'])
        # flowrate [mol/s]
        MoFlRa = self.modelInput['feed']['molar-flowrate']
        # component flowrate [mol/s]
        MoFlRai = MoFlRa*MoFri

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4

        # external heat
        ExHe = self.modelInput['external-heat']
        # cooling temperature [K]
        Tm = ExHe['MeTe']
        # overall heat transfer coefficient [J/s.m2.K]
        U = ExHe['OvHeTrCo']
        # heat transfer area over volume [m2/m3]
        a = 4/ReInDi  # ExHe['EfHeTrAr']

        # var no (Fi,T)
        varNo = compNo + 1

        # initial values
        IV = np.zeros(varNo)
        IV[0:compNo] = MoFlRai
        IV[compNo] = T
        print(f"IV: {IV}")

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted
        # reaction coefficient
        reactionStochCoeff = self.reactionStochCoeffList

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # fun parameters
        FunParam = {
            "compList": compList,
            "const": {
                "CrSeAr": CrSeAr,
                "MoWei": MoWei,
                "StHeRe25": StHeRe25,
                "GaMiVi": GaMiVi
            },
            "ReSpec": ReSpec,
            "ExHe": {
                "OvHeTrCo": U,
                "EfHeTrAr": a,
                "MeTe": Tm
            },
            "reactionRateExpr": reactionRateExpr,
            "constBC1": {
                "MoFri0": MoFri,
                "MoFlRa0": MoFlRa,
                "P0": P,
                "T0": T
            }
        }

        P, compList, StHeRe25, reactionListSorted, reactionStochCoeff, ExHe, CrSeAr

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, ReLe])
        t_span = np.array([0, ReLe])
        times = np.linspace(t_span[0], t_span[1], 100)
        # tSpan = np.linspace(0, rea_L, 25)

        # solver setting
        funSet = PlugFlowReactorClass.modelEquationM1
        paramsSet = (reactionListSorted, reactionStochCoeff,
                     FunParam)

        # ode call
        sol = solve_ivp(funSet,
                        t, IV, method="LSODA", t_eval=times, args=(paramsSet,))

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

    def modelEquationM1(t, y, paramsSet):
        """
            M1 model
            mass and energy balance equations
            modelParameters:
                reactionListSorted: reactant/product and coefficient lists
                reactionStochCoeff: reaction stoichiometric coefficient
                FunParam:
                    compList: component list
                    const
                        CrSeAr: reactor cross sectional area [m^2]
                        MoWei: component molecular weight [g/mol]
                        StHeRe25: standard heat of reaction at 25C [kJ/kmol] | [J/mol]
                        GaMiVi: gas mixture viscosity [Pa.s]
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    reactionRateExpr: reaction rate expression
                    constBC1:
                        MoFri0: mole fraction list
                        MoFlRa0: molar flowrate [mol/s]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]
        """
        # params
        reactionListSorted, reactionStochCoeff, FunParam = paramsSet

        # fun params
        # component symbol list
        comList = FunParam['compList']
        # const ->
        const = FunParam['const']
        # cross-sectional area [m^2]
        CrSeAr = const['CrSeAr']
        # component molecular weight [g/mol]
        MoWei = const['MoWei']
        # standard heat of reaction at 25C [kJ/kmol] | [J/mol]
        StHeRe25 = const['StHeRe25']
        # gas viscosity [Pa.s]
        GaMiVi = const['GaMiVi']
        # reaction no
        reactionListNo = const['reactionListNo']
        # dz [m]
        dz = const['dz']
        # reactor spec ->
        ReSpec = FunParam['ReSpec']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']

        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet mole fraction
        MoFri0 = constBC1['MoFri0']
        # inlet molar flowrate [mol/s]
        MoFlRa0 = constBC1['MoFlRa0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # REVIEW
        # t
        # print(f"t: {t}")
        # components no
        # y: component mole fraction, molar flux, temperature
        compNo = len(comList)
        indexT = compNo

        # molar flowrate list [mol/m^3]
        MoFlRai = y[0:compNo]

        # temperature [K]
        T = y[indexT]

        # FIXME
        # no pressure drop
        # pressure [Pa]
        P = P0

        # total flowrate [mol/m^3]
        MoFlRa = np.sum(MoFlRai)

        # volumetric flowrate [m^3/s]
        VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

        # concentration species [mol/m^3]
        CoSpi = calConcentrationIG(MoFlRai, VoFlRai)

        # mole fraction
        MoFri = rmtUtil.moleFractionFromConcentrationSpecies(CoSpi)

        # kinetics
        # loop
        loopVars0 = (T, P, MoFri, CoSpi)

        # component formation rate [mol/m^3.s]
        # check unit
        RiLoop = np.array(reactionRateExe(
            loopVars0, varisSet, ratesSet))
        Ri = np.copy(RiLoop)

        # component formation rate [mol/m^3.s]
        # ri = np.zeros(compNo)
        # for k in range(compNo):
        #     # reset
        #     _riLoop = 0
        #     for m in range(len(reactionStochCoeff)):
        #         for n in range(len(reactionStochCoeff[m])):
        #             if comList[k] == reactionStochCoeff[m][n][0]:
        #                 _riLoop += reactionStochCoeff[m][n][1]*Ri[m]
        #         ri[k] = _riLoop

        # call [mol/m^3.s]
        ri = componentFormationRate(
            compNo, comList, reactionStochCoeff, Ri)

        # enthalpy
        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
        # Cp mean list
        CpMeanList = calMeanHeatCapacityAtConstantPressure(comList, T)
        # print(f"Cp mean list: {CpMeanList}")
        # Cp mixture
        CpMeanMixture = calMixtureHeatCapacityAtConstantPressure(
            MoFri, CpMeanList)
        # print(f"Cp mean mixture: {CpMeanMixture}")

        # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
        # enthalpy change
        EnChList = np.array(calEnthalpyChangeOfReaction(reactionListSorted, T))
        # heat of reaction at T [kJ/kmol] | [J/mol]
        HeReT = np.array(EnChList + StHeRe25)
        # overall heat of reaction [J/m^3.s]
        OvHeReT = np.dot(Ri, HeReT)

        # cooling temperature [K]
        Tm = ExHe['MeTe']
        # overall heat transfer coefficient [J/s.m2.K]
        U = ExHe['OvHeTrCo']
        # heat transfer area over volume [m2/m3]
        a = ExHe['EfHeTrAr']
        # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
        Ua = U*a
        # external heat [J/m^3.s]
        Qm = Ua*(Tm - T)

        # diff/dt
        dxdt = []
        # loop vars
        const_F1 = 1/CrSeAr
        const_T1 = MoFlRa*CpMeanMixture/CrSeAr

        # mass balance (molar flowrate) [mol/s]
        for i in range(compNo):
            dxdt_F = (1/const_F1)*ri[i]
            dxdt.append(dxdt_F)

        # energy balance (temperature) [K]
        dxdt_T = (1/const_T1)*(-OvHeReT + Qm)
        dxdt.append(dxdt_T)

        return dxdt
