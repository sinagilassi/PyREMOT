# PACKED-BED REACTOR MODEL
# -------------------------

# import packages/modules
import math as MATH
import numpy as np
from numpy.lib import math
from library.plot import plotClass as pltc
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from scipy.optimize import fsolve
from scipy import optimize
# internal
from .modelSetting import MODEL_SETTING, PROCESS_SETTING
from .rmtUtility import rmtUtilityClass as rmtUtil
from .rmtThermo import *
from .fluidFilm import *
from .rmtReaction import reactionRateExe, componentFormationRate
from .gasTransPor import calTest
from core.errors import errGeneralClass as errGeneral
from data.inputDataReactor import *
from core import constants as CONST
from core.utilities import roundNum, selectFromListByIndex
from core.config import REACTION_RATE_ACCURACY
from solvers.solSetting import solverSetting
from core.eqConstants import CONST_EQ_Sh
from solvers.solOrCo import OrCoClass
from solvers.solCatParticle import OrCoCatParticleClass
from solvers.solFiDi import FiDiBuildCMatrix, FiDiBuildTMatrix, FiDiSetMatrix, FiDiBuildCMatrix_DiLe, FiDiBuildTMatrix_DiLe
from solvers.solFiDi import FiDiMeshGenerator, FiDiDerivative1, FiDiDerivative2, FiDiNonUniformDerivative1, FiDiNonUniformDerivative2
from solvers.odeSolver import AdBash3, PreCorr3
from solvers.solResultAnalysis import setOptimizeRootMethod, sortedResult3


class PackedBedReactorClass:
    # def main():
    """
    Packed-bed Reactor Model
    M1 model: packed-bed plug-flow model (1D model)
        assumptions:
            homogeneous
            no dispersion/diffusion along the reactor length
            no radial variation of concentration and temperature
            mass balance is based on flux
            ergun equation is used for pressure drop
            neglecting gravitational effects, kinetic energy, and viscosity change
    M2 model: dynamic homogenous modeling
    M3 model: steady-state homogenous modeling
    """
    # internal data
    _internalData = []

    def __init__(self, modelInput, internalData, reactionListSorted, reactionStochCoeffList):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted
        self.reactionStochCoeffList = reactionStochCoeffList

    # @property
    # def internalData(cls):
    #     return cls._internalData

    # @internalData.setter
    # def internalData(cls, value):
    #     cls._internalData.clear()
    #     cls._internalData.extend(value)

    def runM10(self):
        """
        M1 modeling case
        """

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # ->
        modelParameters = {
            "pressure": P,
            "temperature": T
        }

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
        sol = solve_ivp(PackedBedReactorClass.modelEquationM1,
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

# NOTE
# steady-state homogenous modeling

    def runM1(self):
        """
        M1 modeling case
        steady-state modeling of plug-flow reactor
        unknowns: Fi,F*,T,P
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']

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
        labelList.append("Flux")
        labelList.append("Temperature")
        labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexFlux = compNo
        indexTemp = indexFlux + 1
        indexPressure = indexTemp + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed porosity (bed void fraction)
        BeVoFr = ReSpec['BeVoFr']

        # mole fraction
        MoFri = np.array(self.modelInput['feed']['mole-fraction'])
        # flowrate [mol/s]
        MoFlRa = self.modelInput['feed']['molar-flowrate']
        # component flowrate [mol/s]
        MoFlRai = MoFlRa*MoFri
        # flux [mol/m^2.s]
        MoFl = MoFlRa/(CrSeAr)
        # component flux [mol/m^2.s]
        MoFli = MoFl*MoFri

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']
        # cooling temperature [K]
        Tm = ExHe['MeTe']
        # overall heat transfer coefficient [J/s.m2.K]
        U = ExHe['OvHeTrCo']
        # heat transfer area over volume [m2/m3]
        a = 4/ReInDi  # ExHe['EfHeTrAr']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # var no (Fi,FT,T,P)
        varNo = compNo + 3

        # initial values
        IV = np.zeros(varNo)
        IV[0:compNo] = MoFlRai
        IV[indexFlux] = MoFl
        IV[indexTemp] = T
        IV[indexPressure] = P
        # print(f"IV: {IV}")

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
            "reactionRateExpr": reactionRateExpr
        }

        # save data
        timesNo = solverSetting['S3']['timesNo']

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, ReLe])
        t_span = np.array([0, ReLe])
        times = np.linspace(t_span[0], t_span[1], timesNo)
        # tSpan = np.linspace(0, rea_L, 25)

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # ode solver call
        sol = solve_ivp(PackedBedReactorClass.modelEquationM1,
                        t, IV, method=solverIVP, t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

        # ode result
        successStatus = sol.success
        dataX = sol.t
        # all results
        dataYs = sol.y
        # molar flowrate [mol/s]
        dataYs1 = sol.y[0:compNo, :]
        labelListYs1 = labelList[0:compNo]
        # REVIEW
        # convert molar flowrate [mol/s] to mole fraction
        dataYs1_Ftot = np.sum(dataYs1, axis=0)
        dataYs1_MoFri = dataYs1/dataYs1_Ftot
        # flux
        dataYs2 = sol.y[indexFlux, :]
        labelListYs2 = labelList[indexFlux]
        # temperature
        dataYs3 = sol.y[indexTemp, :]
        labelListYs3 = labelList[indexTemp]
        # pressure
        dataYs4 = sol.y[indexPressure, :]

        # FIXME
        # build matrix
        _dataYs = np.concatenate(
            (dataYs1_MoFri, [dataYs2], [dataYs3], [dataYs4]), axis=0)
        # steady-state results [mole fraction, temperature]
        _ssdataYs = np.concatenate(
            (dataYs1_MoFri, [dataYs3]), axis=0)

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # plot info
        plotTitle = f"Steady-State Modeling [M1] with timesNo: {timesNo} within {elapsed} seconds"

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataX, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexFlux], dataList[indexTemp], dataList[indexPressure]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([0, -2], dataLists)
            # subplot result
            pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
                            "Concentration (mol/m^3)", plotTitle)

            # plot result
            # pltc.plots2D(dataList[0:compNo], "Reactor Length (m)",
            #              "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexFlux], "Reactor Length (m)",
            #              "Flux (kmol/m^2.s)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexTemp], "Reactor Length (m)",
            #              "Temperature (K)", "1D Plug-Flow Reactor")

        else:
            # error
            print(f"Final result: {successStatus}")
            _dataYs = []
            XYList = []
            dataList = []

        # return
        res = {
            "dataYs": _ssdataYs,
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM1(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
            M1 model
            mass, energy, and momentum balance equations
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
                        PARAMS, VARS, RATES

        """
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
        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # bed porosity (bed void fraction)
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexFlux = compNo
        indexT = indexFlux + 1
        indexP = indexT + 1

        # molar flowrate list [mol/m^3]
        MoFlRai = y[0:compNo]
        # total molar flux [mol/m^2.s]
        MoFl = y[indexFlux]
        # temperature [K]
        T = y[indexT]
        # pressure [Pa]
        P = y[indexP]

        # total flowrate [mol/m^3]
        MoFlRa = np.sum(MoFlRai)

        # volumetric flowrate [m^3/s]
        VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

        # concentration species [mol/m^3]
        CoSpi = calConcentrationIG(MoFlRai, VoFlRai)
        # total concentration [mol/m^3]
        CoSp = np.sum(CoSpi)

        # mole fraction
        MoFri = rmtUtil.moleFractionFromConcentrationSpecies(CoSpi)
        # MoFri2 = rmtUtil.moleFractionFromConcentrationSpecies(MoFlRai)

        # interstitial gas velocity [m/s]
        InGaVe = rmtUtil.calSuperficialGasVelocityFromEOS(MoFl, P, T)
        # superficial gas velocity [m/s]
        SuGaVe = InGaVe*BeVoFr

        # mixture molecular weight [kg/mol]
        MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe = calDensityIG(MiMoWe, CoSp)
        GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

        # ergun equation
        ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
        ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
        ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
        ergD = (1-BeVoFr)/(BeVoFr**3)
        RHS_ergun = -1*(ergA*ergB + ergC*ergD)

        # NOTE
        # kinetics
        # component formation rate [mol/m^3.s]
        # conversion
        # FIXME
        # Ri2 = 1000*np.array(PackedBedReactorClass.modelReactions(
        #     P, T, MoFri, CaBeDe))

        # loop
        loopVars0 = (T, P, MoFri, CoSpi)

        # component formation rate [mol/m^3.s]
        # check unit
        RiLoop = np.array(reactionRateExe(
            loopVars0, varisSet, ratesSet))
        Ri = np.copy(RiLoop)

        # component formation rate [mol/m^3.s]
        # rf[mol/kgcat.s]*CaBeDe[kgcat/m^3]
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

        # overall formation rate [mol/m^3.s]
        OvR = np.sum(ri)

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
        Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
            Tm, T, U, a, 'J/m^3.s')

        # diff/dt
        dxdt = []
        # loop vars
        # FIXME
        # const_F1 = 1/(CrSeAr*BeVoFr)
        const_F1 = 1/(CrSeAr)
        const_T1 = MoFl*CpMeanMixture
        const_T2 = MoFlRa*CpMeanMixture/CrSeAr

        # mass balance (molar flowrate) [mol/s]
        for i in range(compNo):
            dxdt_F = (1/const_F1)*ri[i]
            dxdt.append(dxdt_F)
        # flux
        dxdt_Fl = OvR
        dxdt.append(dxdt_Fl)

        # energy balance (temperature) [K]
        dxdt_T = (1/const_T1)*(-OvHeReT + Qm)
        dxdt.append(dxdt_T)

        # momentum balance (ergun equation)
        dxdt_P = RHS_ergun
        dxdt.append(dxdt_P)

        return dxdt

# NOTE
# dynamic homogenous modeling

    def runM2(self):
        """
        M2 modeling case
        dynamic model
        unknowns: Ci, T (dynamic), P (static)
        """
        # NOTE
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        # labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # length list
        dataXs = np.linspace(0, ReLe, zNo)
        # element size - dz [m]
        dz = ReLe/(zNo-1)

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # total var no along the reactor length
        varNoT = varNo*zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (varNo, zNo)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for i in range(compNo):
            for j in range(zNo):
                IV2D[i][j] = SpCoi0[i]

        for j in range(zNo):
            IV2D[indexTemp][j] = T

        # flatten IV
        IV = IV2D.flatten()

        # print(f"IV: {IV}")

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
                "GaMiVi": GaMiVi,
                "zNo": zNo,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
                "dz": dz
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "reactionRateExpr": reactionRateExpr,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T
            }
        }

        # time span
        tNo = solverSetting['S2']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['S2']['timesNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((varNo, tNo, zNo))

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)
            print(f"time: {t} seconds")

            # ode call
            sol = solve_ivp(PackedBedReactorClass.modelEquationM2,
                            t, IV, method=solverIVP, t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

            # ode result
            successStatus = sol.success
            # check
            if successStatus is False:
                raise

            # time interval
            dataTime = sol.t
            # all results
            dataYs = sol.y

            # component concentration [mol/m^3]
            dataYs1 = dataYs[0:varNoCon, -1]
            # 2d matrix
            dataYs1_Reshaped = np.reshape(dataYs1, (compNo, zNo))
            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1_Reshaped, axis=0)
            dataYs1_MoFri = dataYs1_Reshaped/dataYs1_Ctot
            # temperature - 2d matrix
            dataYs2 = np.array([dataYs[varNoCon:varNoT, -1]])

            # combine
            _dataYs = np.concatenate((dataYs1_MoFri, dataYs2), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCons": dataYs1_Reshaped,
                "dataYTemp": dataYs2,
                "dataYs": _dataYs
            })

            for m in range(varNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # NOTE
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        # ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        # ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        # ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        # ssdataLists = [ssdataList[0:compNo],
        #                ssdataList[indexTemp]]
        # subplot result
        # pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        # plot info
        plotTitle = f"Dynamic Modeling [M2] for opT: {opT} with zNo: {zNo}, tNo: {tNo}"

        # REVIEW
        # display result at specific time
        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataYs']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(dataLists, "Reactor Length (m)",
                                "Concentration (mol/m^3)", plotTitle)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _name = labelList[i] + " at t=" + str(opTSpan[t+1])

                _labelNameTime.append(_name)

            dataList = pltc.plots2DSetDataList(XYList, _labelNameTime)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select items
        # indices = [0, 2, -1]
        # selected_elements = [_dataListsLoop[index] for index in indices]
        # select datalist
        _dataListsSelected = selectFromListByIndex([1, -1], _dataListsLoop)

        # subplot result
        # pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM2(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
            M2 model [dynamic modeling]
            mass, energy, and momentum balance equations
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
                        zNo: number of finite difference in the z direction
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                        dz: differential length [m]
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    reactionRateExpr: reaction rate expression
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]

        """
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
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # zNo
        zNo = const['zNo']
        # var no.
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # calculate
        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # superficial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate
        Ri_z = np.zeros((zNo, reactionListNo))

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1

        # species concentration [kmol/m^3]
        CoSpi = np.zeros(compNo)

        # reaction rate
        ri = np.zeros(compNo)

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (varNo, zNo))

        # -> concentration [mol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        for i in range(compNo):
            _SpCoi = yLoop[i, :]
            SpCoi_z[i, :] = _SpCoi

        # temperature [K]
        T_z = np.zeros(zNo)
        T_z = yLoop[indexT, :]

        # diff/dt
        # dxdt = []
        # matrix
        dxdtMat = np.zeros((varNo, zNo))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(zNo):
            ## block ##

            # FIXME
            # concentration species [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = max(_SpCoi_z, CONST.EPS_CONST)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)

            # temperature [K]
            T = T_z[z]
            # pressure [Pa]
            P = P_z[z]

            ## calculate ##
            # mole fraction
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

            # gas velocity based on interstitial velocity [m/s]
            InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            SuGaVe = InGaVe*BeVoFr

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = CoSp*SuGaVe*CrSeAr
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp)
            GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # NOTE
            # REVIEW
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaBeDe[kgcat/m^3]
            # SpCoi conversion
            _SpCoi = 1e3*CoSpi
            # loop
            loopVars0 = (T, P, MoFri, _SpCoi)
            # check unit
            RiLoop = 1e-3*np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))
            Ri_z[z, :] = RiLoop

            # REVIEW
            # component formation rate [kmol/m^3.s]
            ri = componentFormationRate(
                compNo, comList, reactionStochCoeff, Ri_z[z, :])

            # overall formation rate [kmol/m^3.s]
            OvR = np.sum(ri)

            # NOTE
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
            EnChList = np.array(
                calEnthalpyChangeOfReaction(reactionListSorted, T))
            # heat of reaction at T [kJ/kmol] | [J/mol]
            HeReT = np.array(EnChList + StHeRe25)
            # overall heat of reaction [kJ/m^3.s]
            # exothermic reaction (negative sign)
            # endothermic sign (positive sign)
            OvHeReT = np.dot(Ri_z[z, :], HeReT)

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m2/m3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            Ua = U*a
            # external heat [kJ/m^3.s]
            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T, U, a, 'kJ/m^3.s')

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # loop vars
            const_F1 = 1/BeVoFr
            const_T1 = MoFl*CpMeanMixture
            const_T2 = 1/(CoSp*CpMeanMixture*BeVoFr + (1-BeVoFr)*CaDe*CaSpHeCa)

            # NOTE
            # concentration [mol/m^3]
            for i in range(compNo):
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # check BC
                if z == 0:
                    # BC1
                    Ci_b = SpCoi0[i]
                else:
                    # interior nodes
                    Ci_b = max(SpCoi_z[i][z - 1], CONST.EPS_CONST)
                # backward difference
                dCdz = (Ci_c - Ci_b)/dz
                # mass balance
                dxdt_F = const_F1*(-SuGaVe*dCdz + ri[i])
                dxdtMat[i][z] = dxdt_F

            # energy balance (temperature) [K]
            # temp [K]
            T_c = T_z[z]
            # check BC
            if z == 0:
                # BC1
                T_b = T0
            else:
                # interior nodes
                T_b = T_z[z - 1]
            # backward difference
            dTdz = (T_c - T_b)/dz

            dxdt_T = const_T2*(-const_T1*dTdz + (-OvHeReT + Qm))
            dxdtMat[indexT][z] = dxdt_T

        # flat
        dxdt = dxdtMat.flatten().tolist()

        print("time: ", t)
        return dxdt

# NOTE
# steady-state homogenous modeling

    def runM3(self):
        """
        M3 modeling case
        steady-state modeling
            not exactly plug-flow as dv/dz = 0
        unknowns: Ci, T, P
            velocity is calculated from EOS consiering feed Tf, Pf, Cf
        """
        # NOTE
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']

        # model info
        modelId = self.modelInput['model']
        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # REVIEW
        # inlet species concentration [mol/m^3]
        SpCoi0 = 1*np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [mol/m^3]
        SpCo0 = np.sum(SpCoi0)

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # var no (Ci,T,P)
        varNo = compNo + 2

        # initial values
        IV = np.zeros(varNo)
        IV[0:compNo] = SpCoi0
        IV[indexTemp] = T
        IV[indexPressure] = P
        # print(f"IV: {IV}")

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
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T
            },
            "reactionRateExpr": reactionRateExpr,
        }

        # save data
        # timesNo = solverSetting['S3']['timesNo']
        timesNo = solverSetting['M9']['zNo']

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, ReLe])
        t_span = np.array([0, ReLe])
        times = np.linspace(t_span[0], t_span[1], timesNo)
        # tSpan = np.linspace(0, rea_L, 25)

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # ode call
        sol = solve_ivp(PackedBedReactorClass.modelEquationM3,
                        t, IV, method=solverIVP, t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

        # ode result
        successStatus = sol.success
        dataX = sol.t
        # all results
        dataYs = sol.y
        # concentration [mol/m^3]
        dataYs1 = sol.y[0:compNo, :]
        labelListYs1 = labelList[0:compNo]
        # REVIEW
        # convert molar flowrate to mole fraction
        # convert concentration to mole fraction
        dataYs1_Ctot = np.sum(dataYs1, axis=0)
        dataYs1_MoFri = dataYs1/dataYs1_Ctot
        # temperature
        dataYs2 = sol.y[indexTemp, :]
        labelListYs3 = labelList[indexTemp]
        # pressure
        dataYs3 = sol.y[indexPressure, :]

        # FIXME
        # build matrix
        _dataYs = np.concatenate(
            (dataYs1_MoFri, [dataYs2]), axis=0)

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # plot info
        plotTitle = f"Steady-State Modeling {modelId} with timesNo: {timesNo} within {elapsed}"

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataX, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([0, -1], dataLists)
            # subplot result
            pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
                            "Concentration (mol/m^3)", plotTitle)

            # plot result
            # pltc.plots2D(dataList[0:compNo], "Reactor Length (m)",
            #              "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexFlux], "Reactor Length (m)",
            #              "Flux (kmol/m^2.s)", "1D Plug-Flow Reactor")

            # pltc.plots2D(dataList[indexTemp], "Reactor Length (m)",
            #              "Temperature (K)", "1D Plug-Flow Reactor")

        else:
            _dataYs = []
            XYList = []
            dataList = []

        # return
        res = {
            "dataYs": _dataYs,
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM3(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
        M3 model
        mass, energy, and momentum balance equations
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
            reactionRateExpr: reaction rate expressions
        """
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
        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # exchange heat spec ->
        ExHe = FunParam['ExHe']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [mol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [mol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # calculate
        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1

        # concentration species [mol/m^3]
        CoSpi = y[0:compNo]
        # temperature [K]
        T = y[indexT]
        # pressure [Pa]
        P = y[indexP]

        # total concentration [mol/m^3]
        CoSp = np.sum(CoSpi)

        # mole fraction
        MoFri = np.array(
            rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

        # gas velocity based on interstitial velocity [m/s]
        InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
        # superficial gas velocity [m/s]
        SuGaVe = InGaVe*BeVoFr

        # total flowrate [mol/s]
        # [mol/m^3]*[m/s]*[m^2]
        MoFlRa = CoSp*SuGaVe*CrSeAr
        # molar flowrate list [mol/s]
        MoFlRai = MoFlRa*MoFri

        # FIXME
        # molar flux [mol/m^2.s]
        MoFl = MoFlRa/CrSeAr

        # volumetric flowrate [m^3/s]
        VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

        # mixture molecular weight [kg/mol]
        MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe = calDensityIG(MiMoWe, CoSp)
        GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

        # NOTE
        # momentum equation
        # REVIEW
        # ergun equation
        ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
        ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
        ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
        ergD = (1-BeVoFr)/(BeVoFr**3)
        RHS_ergun = -1*(ergA*ergB + ergC*ergD)

        # NOTE
        # kinetics
        # component formation rate [mol/m^3.s]
        # conversion
        # FIXME
        # Ri = 1000*np.array(PackedBedReactorClass.modelReactions(
        #     P, T, MoFri, CaBeDe))
        # loop
        loopVars0 = (T, P, MoFri, CoSpi)
        # check unit
        r0 = np.array(reactionRateExe(
            loopVars0, varisSet, ratesSet))

        # loop
        Ri = r0

        # component formation rate [mol/m^3.s]
        # rf[mol/kgcat.s]*CaBeDe[kgcat/m^3]
        # call [mol/m^3.s]
        ri = componentFormationRate(
            compNo, comList, reactionStochCoeff, Ri)

        # overall formation rate [mol/m^3.s]
        OvR = np.sum(ri)

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

        # NOTE
        #
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

        # NOTE
        # diff/dt
        dxdt = []
        # loop vars
        const_C1 = 1/SuGaVe
        const_T1 = 1/(MoFl*CpMeanMixture)

        # mass balance (concentration) [mol/m^3]
        for i in range(compNo):
            dxdt_Ci = const_C1*ri[i]
            dxdt.append(dxdt_Ci)

        # energy balance (temperature) [K]
        dxdt_T = const_T1*(-OvHeReT + Qm)
        dxdt.append(dxdt_T)

        # momentum balance (ergun equation)
        dxdt_P = RHS_ergun
        dxdt.append(dxdt_P)

        return dxdt

# NOTE
# steady-state homogenous modeling

    def runM4(self):
        """
        M4 modeling case
        steady-state modeling
        unknowns: Ci,P,T,v
            CT, GaDe, are calculated from EOS
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
        labelList.append("Pressure")
        labelList.append("Velocity")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1
        indexDensity = indexVelocity + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [mol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [mol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']

        # mole fraction
        MoFri = np.array(
            rmtUtil.moleFractionFromConcentrationSpecies(SpCoi0))

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # mixture molecular weight [kg/mol]
        MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

        # inlet density [kg/m^3]
        GaDe0 = MiMoWe*SpCo0

        # var no Ci,T,P,v)
        varNo = compNo + 3

        # initial values
        IV = np.zeros(varNo)
        IV[0:compNo] = SpCoi0
        IV[indexTemp] = T
        IV[indexPressure] = P
        IV[indexVelocity] = SuGaVe0
        # print(f"IV: {IV}")

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
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T
            }
        }

        # save data
        timesNo = solverSetting['S3']['timesNo']

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, ReLe])
        t_span = np.array([0, ReLe])
        times = np.linspace(t_span[0], t_span[1], timesNo)
        # tSpan = np.linspace(0, rea_L, 25)

        # ode call
        sol = solve_ivp(PackedBedReactorClass.modelEquationM4,
                        t, IV, method="LSODA", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

        # ode result
        successStatus = sol.success
        dataX = sol.t
        # all results
        dataYs = sol.y
        # concentration [mol/m^3]
        dataYs1 = sol.y[0:compNo, :]
        labelListYs1 = labelList[0:compNo]
        # REVIEW
        # convert concentration to mole fraction
        dataYs1_Ctot = np.sum(dataYs1, axis=0)
        dataYs1_MoFri = dataYs1/dataYs1_Ctot
        # temperature [K]
        dataYs2 = sol.y[indexTemp, :]
        labelListYs3 = labelList[indexTemp]
        # pressure [Pa]
        dataYs3 = sol.y[indexPressure, :]
        # velocity [m/s]
        dataYs4 = sol.y[indexVelocity, :]

        # FIXME
        # build matrix
        _dataYs = np.concatenate(
            (dataYs1_MoFri, [dataYs2]), axis=0)
        _dataYsPlot = np.concatenate(
            (dataYs1_MoFri, [dataYs2], [dataYs3], [dataYs4]), axis=0)

        # plot info
        plotTitle = f"Steady-State Modeling [M4] with timesNo: {timesNo}"

        # NOTE
        # # steady-state result
        # # txt
        # # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # # binary
        # ssModelingResult = np.load('ResM1.npy')
        # # ssdataXs = np.linspace(0, ReLe, zNo)
        # ssXYList = pltc.plots2DSetXYList(dataX, ssModelingResult)
        # ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # # datalists
        # ssdataLists = [ssdataList[0:compNo],
        #                ssdataList[indexTemp]]

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataX, _dataYsPlot)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp], dataList[indexPressure], dataList[indexVelocity]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([0, -3], dataLists)
            # subplot result
            pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
                            "Concentration (mol/m^3)", plotTitle)

        else:
            _dataYs = []
            XYList = []
            dataList = []

        # return
        res = {
            "dataYs": _dataYs,
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM4(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
            M4 model
            mass, energy, and momentum balance equations
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

        """
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
        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # exchange heat spec ->
        ExHe = FunParam['ExHe']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # calculate
        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexVelocity = indexP + 1

        # concentration species [mol/m^3]
        CoSpi = y[0:compNo]
        # temperature [K]
        T = y[indexT]
        # pressure [Pa]
        P = y[indexP]
        # velocity
        SuGaVe = y[indexVelocity]

        # total concentration [mol/m^3]
        CoSp = np.sum(CoSpi)

        # mole fraction
        MoFri = np.array(
            rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

        # gas velocity based on interstitial velocity [m/s]
        # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
        # superficial gas velocity [m/s]
        # SuGaVe = InGaVe*BeVoFr

        # total flowrate [mol/s]
        # [mol/m^3]*[m/s]*[m^2]
        MoFlRa = CoSp*SuGaVe*CrSeAr
        # molar flowrate list [mol/s]
        MoFlRai = MoFlRa*MoFri

        # molar flux [mol/m^2.s]
        MoFl = MoFlRa/CrSeAr

        # volumetric flowrate [m^3/s]
        VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

        # mixture molecular weight [kg/mol]
        MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe = calDensityIG(MiMoWe, CoSp)
        # GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

        # NOTE
        # momentum equation
        # REVIEW
        # ergun equation
        ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
        ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
        ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
        ergD = (1-BeVoFr)/(BeVoFr**3)
        RHS_ergun = -1*(ergA*ergB + ergC*ergD)

        # NOTE
        # kinetics
        # component formation rate [mol/m^3.s]
        # conversion
        # FIXME
        Ri = 1000*np.array(PackedBedReactorClass.modelReactions(
            P, T, MoFri, CaBeDe))

        # component formation rate [mol/m^3.s]
        # rf[mol/kgcat.s]*CaBeDe[kgcat/m^3]
        ri = np.zeros(compNo)
        for k in range(compNo):
            # reset
            _riLoop = 0
            for m in range(len(reactionStochCoeff)):
                for n in range(len(reactionStochCoeff[m])):
                    if comList[k] == reactionStochCoeff[m][n][0]:
                        _riLoop += reactionStochCoeff[m][n][1]*Ri[m]
                ri[k] = _riLoop

        # overall formation rate [mol/m^3.s]
        OvR = np.sum(ri)

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

        # NOTE
        #
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

        # REVIEW
        # subs df/dt

        # NOTE
        # diff/dt
        dxdt = []
        # loop vars
        const_C1 = 1/SuGaVe
        const_T1 = 1/(MoFl*CpMeanMixture)
        const_V1 = 1/CoSp

        # RHS of ODE
        # energy balance
        dxdt_T = const_T1*(-OvHeReT + Qm)
        # momentum balance (ergun eq.)
        dxdt_P = RHS_ergun
        # velocity from global concentration
        dxdt_v = const_V1*((-SuGaVe/CONST.R_CONST) *
                           ((1/T)*dxdt_P - (P/T**2)*dxdt_T) + OvR)

        # mass balance (concentration) [mol/m^3]
        for i in range(compNo):
            dxdt_Ci = const_C1*(-CoSpi[i]*dxdt_v + ri[i])
            dxdt.append(dxdt_Ci)

        # energy balance (temperature) [K]
        # dxdt_T = const_T1*(-OvHeReT + Qm)
        dxdt.append(dxdt_T)

        # momentum balance (ergun equation)
        # dxdt_P = RHS_ergun
        dxdt.append(dxdt_P)

        # velocity [m/s]
        dxdt.append(dxdt_v)

        return dxdt

# NOTE
# dynamic homogenous modeling

    def runM5(self):
        """
        M5 modeling case
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static)
            CT, GaDe = f(P, T, n)
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        # labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']

        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # length list
        dataXs = np.linspace(0, ReLe, zNo)
        # element size - dz [m]
        dz = ReLe/(zNo-1)

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # total var no along the reactor length
        varNoT = varNo*zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (varNo, zNo)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for i in range(compNo):
            for j in range(zNo):
                IV2D[i][j] = SpCoi0[i]

        for j in range(zNo):
            IV2D[indexTemp][j] = T

        # flatten IV
        IV = IV2D.flatten()

        # print(f"IV: {IV}")

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
                "GaMiVi": GaMiVi,
                "zNo": zNo,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
                "dz": dz
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T,
                "SuGaVe0": SuGaVe0
            },
            "reactionRateExpr": reactionRateExpr

        }

        # time span
        tNo = solverSetting['S2']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['S2']['timesNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((varNo, tNo, zNo))
        #

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)
            print(f"time: {t} seconds")

            # ode call
            sol = solve_ivp(PackedBedReactorClass.modelEquationM5,
                            t, IV, method=solverIVP, t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

            # ode result
            successStatus = sol.success
            # check
            if successStatus is False:
                raise

            # time interval
            dataTime = sol.t
            # all results
            dataYs = sol.y

            # component concentration [kmol/m^3]
            dataYs1 = dataYs[0:varNoCon, -1]
            # 2d matrix
            dataYs1_Reshaped = np.reshape(dataYs1, (compNo, zNo))
            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1_Reshaped, axis=0)
            dataYs1_MoFri = dataYs1_Reshaped/dataYs1_Ctot
            # temperature - 2d matrix
            dataYs2 = np.array([dataYs[varNoCon:varNoT, -1]])

            # combine
            _dataYs = np.concatenate((dataYs1_MoFri, dataYs2), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCons": dataYs1_Reshaped,
                "dataYTemp": dataYs2,
                "dataYs": _dataYs
            })

            for m in range(varNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # NOTE
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        # ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        # ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        # ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        # ssdataLists = [ssdataList[0:compNo],
        #                ssdataList[indexTemp]]
        # subplot result
        # pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        # plot info
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # display result at specific time
        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataYs']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(dataLists, "Reactor Length (m)",
                                "Concentration (mol/m^3)", plotTitle)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _name = labelList[i] + " at t=" + str(opTSpan[t+1])

                _labelNameTime.append(_name)

            dataList = pltc.plots2DSetDataList(XYList, _labelNameTime)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select items
        # indices = [0, 2, -1]
        # selected_elements = [_dataListsLoop[index] for index in indices]
        # select datalist
        _dataListsSelected = selectFromListByIndex([1, -1], _dataListsLoop)

        # subplot result
        # pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM5(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
            [dynamic modeling]
            mass, energy, and momentum balance equations
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
                        zNo: number of finite difference in the z direction
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                        dz: differential length [m]
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K],
                    reactionRateExpr: reaction rate expressions
                        VARS: list of variable
                        RATES: list of rate expressions
        """
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
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # zNo
        zNo = const['zNo']
        # var no.
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # calculate
        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # superficial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate
        Ri_z = np.zeros((zNo, reactionListNo))

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # superficial gas velocity [m/s]
        v_z = np.zeros(zNo + 1)
        v_z[0] = SuGaVe0

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexV = indexP + 1

        # species concentration [kmol/m^3]
        CoSpi = np.zeros(compNo)

        # reaction rate
        ri = np.zeros(compNo)
        ri0 = np.zeros(compNo)

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (varNo, zNo))

        # -> concentration [mol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        for i in range(compNo):
            _SpCoi = yLoop[i, :]
            SpCoi_z[i, :] = _SpCoi

        # temperature [K]
        T_z = np.zeros(zNo)
        T_z = yLoop[indexT, :]

        # diff/dt
        # dxdt = []
        # matrix
        dxdtMat = np.zeros((varNo, zNo))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(zNo):
            ## block ##

            # FIXME
            # concentration species [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = max(_SpCoi_z, CONST.EPS_CONST)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)

            # temperature [K]
            T = T_z[z]
            # pressure [Pa]
            P = P_z[z]

            # velocity
            v = v_z[z]

            ## calculate ##
            # mole fraction
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

            # TODO
            # dv/dz
            # gas velocity based on interstitial velocity [m/s]
            # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            # SuGaVe = InGaVe*BeVoFr
            # from ode eq. dv/dz
            SuGaVe = v

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = CoSp*SuGaVe*CrSeAr
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp)
            GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # NOTE
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaBeDe[kgcat/m^3]
            # r0 = np.array(PackedBedReactorClass.modelReactions(
            #     P_z[z], T_z[z], MoFri, CaBeDe))

            # loop
            loopVars0 = (T_z[z], P_z[z], MoFri, CoSpi)
            # check unit
            r0 = np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))
            # r0 = np.copy(RiLoop)

            # loop
            Ri_z[z, :] = r0

            # REVIEW
            # component formation rate [kmol/m^3.s]
            # call
            ri = componentFormationRate(
                compNo, comList, reactionStochCoeff, Ri_z[z, :])

            # overall formation rate [kmol/m^3.s]
            OvR = np.sum(ri)

            # NOTE
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
            EnChList = np.array(
                calEnthalpyChangeOfReaction(reactionListSorted, T))
            # heat of reaction at T [kJ/kmol] | [J/mol]
            HeReT = np.array(EnChList + StHeRe25)
            # overall heat of reaction [kJ/m^3.s]
            # exothermic reaction (negative sign)
            # endothermic sign (positive sign)
            OvHeReT = np.dot(Ri_z[z, :], HeReT)

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m2/m3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            Ua = U*a
            # external heat [kJ/m^3.s]
            # if Tm == 0:
            #     # adiabatic
            #     Qm0 = 0
            # else:
            #     # heat added/removed from the reactor
            #     # Tm > T: heat is added (positive sign)
            #     # T > Tm: heat removed (negative sign)
            #     Qm0 = (Ua*(Tm - T))*1e-3

            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T, U, a, 'kJ/m^3.s')

            # NOTE
            # velocity from global concentration
            # check BC
            if z == 0:
                # BC1
                T_b = T0
            else:
                # interior nodes
                T_b = T_z[z - 1]

            dxdt_v_T = (T_z[z] - T_b)/dz
            # CoSp x 1000
            # OvR x 1000
            dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
                                      ((1/T)*dxdt_P - (P/T**2)*dxdt_v_T) + OvR*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            v_z[z+1] = dxdt_v*dz + v_z[z]

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # loop vars
            const_F1 = 1/BeVoFr
            const_T1 = MoFl*CpMeanMixture
            const_T2 = 1/(CoSp*CpMeanMixture*BeVoFr + (1-BeVoFr)*CaDe*CaSpHeCa)

            # NOTE

            # concentration [mol/m^3]
            for i in range(compNo):
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # check BC
                if z == 0:
                    # BC1
                    Ci_b = SpCoi0[i]
                else:
                    # interior nodes
                    Ci_b = max(SpCoi_z[i][z - 1], CONST.EPS_CONST)
                # backward difference
                dCdz = (Ci_c - Ci_b)/dz
                # mass balance
                dxdt_F = const_F1*(-v_z[z]*dCdz - Ci_c*dxdt_v + ri[i])
                dxdtMat[i][z] = dxdt_F

            # energy balance (temperature) [K]
            # temp [K]
            T_c = T_z[z]
            # check BC
            if z == 0:
                # BC1
                T_b = T0
            else:
                # interior nodes
                T_b = T_z[z - 1]
            # backward difference
            dTdz = (T_c - T_b)/dz

            dxdt_T = const_T2*(-const_T1*dTdz + (-OvHeReT + Qm))
            dxdtMat[indexT][z] = dxdt_T

        # flat
        dxdt = dxdtMat.flatten().tolist()
        print("time: ", t)

        return dxdt

# NOTE
#! dynamic heterogenous modeling

    def runM6(self):
        """
        M6 modeling case
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static), Cci, Tc (dynamic, for catalyst)
            CT, GaDe = f(P, T, n)
        numerical method: orthogonal collocation
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        # labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # REVIEW
        # domain length
        DoLe = 1
        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # length list
        dataXs = np.linspace(0, ReLe, zNo)
        # element size - dz [m]
        dz = ReLe/(zNo-1)
        # orthogonal collocation points in the r direction
        rNo = solverSetting['S2']['rNo']

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total number
        varNoConInSolid = varNoConInSolidBlock*zNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of layers
        noLayer = compNo + 1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = rNo + 1
        # number of var columns [i]
        varNoColumns = zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoRows, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayer - 1):
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # gas phase
                        IV2D[m][j][i] = SpCoi0[m]
                    else:
                        # solid phase
                        IV2D[m][j][i] = SpCoi0[m]

        # temperature
        for i in range(varNoColumns):
            for j in range(varNoRows):
                # separate phase
                if j == 0:
                    # gas phase
                    IV2D[noLayer - 1][j][i] = T
                else:
                    # solid phase
                    IV2D[noLayer - 1][j][i] = T

        # flatten IV
        IV = IV2D.flatten()

        # print(f"IV: {IV}")

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted
        # reaction coefficient
        reactionStochCoeff = self.reactionStochCoeffList

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # FIXME
        # solver setting
        # orthogonal collocation method
        OrCoClassSet = OrCoClass()
        OrCoClassSetRes = OrCoClassSet.buildMatrix()

        # fun parameters
        FunParam = {
            "compList": compList,
            "const": {
                "CrSeAr": CrSeAr,
                "MoWei": MoWei,
                "StHeRe25": StHeRe25,
                "GaMiVi": GaMiVi,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T,
                "SuGaVe0": SuGaVe0
            },
            "meshSetting": {
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz
            },
            "solverSetting": {
                "OrCoClassSetRes": OrCoClassSetRes
            },
            "reactionRateExpr": reactionRateExpr

        }

        # time span
        tNo = solverSetting['S2']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['S2']['timesNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((varNo, tNo, zNo))
        #

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            # method [1]: LSODA, [2]: BDF, [3]: Radau
            sol = solve_ivp(PackedBedReactorClass.modelEquationM6,
                            t, IV, method=solverIVP, t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

            # ode result
            successStatus = sol.success
            # check
            if successStatus is False:
                raise

            # time interval
            dataTime = sol.t
            # all results
            # components, temperature layers
            dataYs = sol.y

            # std format
            dataYs_Reshaped = np.reshape(
                dataYs[:, -1], (noLayer, varNoRows, varNoColumns))

            # component concentration [kmol/m^3]
            # Ci and Cs
            # dataYs1 = dataYs[0:varNoCon, -1]
            # 3d matrix
            # dataYs1_Reshaped = np.reshape(
            #     dataYs1, (compNo, varNoRows, varNoColumns))

            dataYs1_Reshaped = dataYs_Reshaped[:-1]

            # gas phase
            dataYs1GasPhase = dataYs1_Reshaped[:, 0, :]
            # solid phase
            dataYs1SolidPhase = dataYs1_Reshaped[:, 1:, :]

            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1GasPhase, axis=0)
            dataYs1_MoFri = dataYs1GasPhase/dataYs1_Ctot

            # temperature - 2d matrix
            # dataYs2 = np.array([dataYs[varNoCon:varNoLayerT, -1]])
            # 2d matrix
            # dataYs2_Reshaped = np.reshape(
            #     dataYs2, (1, varNoRows, varNoColumns))

            dataYs2_Reshaped = dataYs_Reshaped[indexTemp]
            # gas phase
            dataYs2GasPhase = dataYs2_Reshaped[0, :].reshape((1, zNo))
            # solid phase
            dataYs2SolidPhase = dataYs2_Reshaped[1:, :]

            # combine
            _dataYs = np.concatenate(
                (dataYs1_MoFri, dataYs2GasPhase), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCon": dataYs1GasPhase,
                "dataYTemp": dataYs2GasPhase,
                "dataYs": _dataYs,
                "dataYCons": dataYs1SolidPhase,
                "dataYTemps": dataYs2SolidPhase,
            })

            for m in range(varNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # NOTE
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        ssdataLists = [ssdataList[0:compNo],
                       ssdataList[indexTemp]]
        # subplot result
        # pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        # plot info
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # display result at specific time
        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataYs']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(dataLists, "Reactor Length (m)",
                                "Concentration (mol/m^3)", plotTitle, ssdataLists)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _name = labelList[i] + " at t=" + str(opTSpan[t+1])

                _labelNameTime.append(_name)

            dataList = pltc.plots2DSetDataList(XYList, _labelNameTime)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select items
        # indices = [0, 2, -1]
        # selected_elements = [_dataListsLoop[index] for index in indices]
        # select datalist
        _dataListsSelected = selectFromListByIndex([1, -1], _dataListsLoop)

        # subplot result
        # pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM6(t, y, reactionListSorted, reactionStochCoeff, FunParam):
        """
            M6 model [dynamic modeling]
            mass, energy, and momentum balance equations
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
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]
                    meshSetting:
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                    solverSetting:
                        OrCoClassSetRes: constants of OC methods
                    reactionRateExpr: reaction rate expressions

        """
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

        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']
        # catalyst porosity
        CaPo = ReSpec['CaPo']
        # catalyst tortuosity
        CaTo = ReSpec['CaTo']
        # catalyst thermal conductivity [J/K.m.s]
        CaThCo = ReSpec['CaThCo']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']

        # mesh setting
        meshSetting = FunParam['meshSetting']
        # number of layers
        noLayer = meshSetting['noLayer']
        # var no in each layer
        varNoLayer = meshSetting['varNoLayer']
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = meshSetting['varNoLayerT']
        # number of var rows [j]
        varNoRows = meshSetting['varNoRows']
        # number of var columns [i]
        varNoColumns = meshSetting['varNoColumns']
        # rNo
        rNo = meshSetting['rNo']
        # zNo
        zNo = meshSetting['zNo']
        # dz [m]
        dz = meshSetting['dz']

        # solver setting
        solverSetting = FunParam['solverSetting']
        # number of collocation points
        ocN = solverSetting['OrCoClassSetRes']['N']
        ocXc = solverSetting['OrCoClassSetRes']['Xc']
        ocA = solverSetting['OrCoClassSetRes']['A']
        ocB = solverSetting['OrCoClassSetRes']['B']
        ocQ = solverSetting['OrCoClassSetRes']['Q']

        # init OrCoCatParticle
        OrCoCatParticleClassSet = OrCoCatParticleClass(
            ocXc, ocN, ocQ, ocA, ocB, varNo)

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexV = indexP + 1

        # calculate
        # particle radius
        PaRa = PaDi/2
        # specific surface area exposed to the free fluid [m^2/m^3]
        SpSuAr = (3/PaRa)*(1 - BeVoFr)

        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # interstitial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate in the solid phase
        Ri_z = np.zeros((zNo, reactionListNo))
        Ri_zr = np.zeros((zNo, rNo, reactionListNo))
        Ri_r = np.zeros((rNo, reactionListNo))
        # reaction rate
        # ri = np.zeros(compNo) # deprecate
        # ri0 = np.zeros(compNo) # deprecate
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        # overall reaction
        OvR = np.zeros(rNo)
        # overall enthalpy
        OvHeReT = np.zeros(rNo)
        # heat capacity at constant pressure
        SoCpMeanMix = np.zeros(rNo)

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # superficial gas velocity [m/s]
        v_z = np.zeros(zNo + 1)
        v_z[0] = SuGaVe0

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (noLayer, varNoRows, varNoColumns))

        # all species concentration in gas & solid phase
        SpCo_mz = np.zeros((noLayer - 1, varNoRows, varNoColumns))
        # all species concentration in gas phase [kmol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        # all species concentration in solid phase (catalyst) [kmol/m^3]
        SpCosi_mzr = np.zeros((compNo, rNo, zNo))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCo_mz[m] = _SpCoi
        # concentration in the gas phase [kmol/m^3]
        for m in range(compNo):
            for j in range(varNoRows):
                if j == 0:
                    # gas phase
                    SpCoi_z[m, :] = SpCo_mz[m, j, :]
                else:
                    # solid phase
                    SpCosi_mzr[m, j-1, :] = SpCo_mz[m, j, :]

        # species concentration in gas phase [kmol/m^3]
        CoSpi = np.zeros(compNo)
        # total concentration [kmol/m^3]
        CoSp = 0
        # species concentration in solid phase (catalyst) [kmol/m^3]
        # shape
        CosSpiMatShape = (rNo, compNo)
        CosSpi_r = np.zeros(CosSpiMatShape)
        # total concentration in the solid phase [kmol/m^3]
        CosSp_r = np.zeros(rNo)

        # flux
        MoFli_z = np.zeros(compNo)

        # NOTE
        # temperature [K]
        T_mz = np.zeros((varNoRows, varNoColumns))
        T_mz = yLoop[noLayer - 1]
        # temperature in the gas phase
        T_z = np.zeros(zNo)
        T_z = T_mz[0, :]
        # temperature in solid phase
        Ts_z = np.zeros((rNo, zNo))
        Ts_z = T_mz[1:]
        # temperature in the solid phase
        Ts_r = np.zeros(rNo)

        # diff/dt
        # dxdt = []
        # matrix
        # dxdtMat = np.zeros((varNo, zNo))
        dxdtMat = np.zeros((noLayer, varNoRows, varNoColumns))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(varNoColumns):
            ## block ##

            # concentration species in the gas phase [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = max(_SpCoi_z, CONST.EPS_CONST)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)

            # FIXME
            # concentration species in the solid phase [kmol/m^3]
            # display concentration list in each oc point (rNo)
            for i in range(compNo):
                for r in range(rNo):
                    _CosSpi_z = SpCosi_mzr[i][r][z]
                    CosSpi_r[r][i] = max(_CosSpi_z, CONST.EPS_CONST)

            # total concentration in the solid phase [kmol/m^3]
            CosSp_r = np.sum(CosSpi_r, axis=1).reshape((rNo, 1))

            # concentration in the outer surface of the catalyst [kmol/m^3]
            CosSpi_cat = CosSpi_r[0]

            # temperature [K]
            T = T_z[z]
            # temperature in the solid phase (for each point)
            # Ts[3], Ts[2], Ts[1], Ts[0]
            Ts_r = Ts_z[:, z]

            # pressure [Pa]
            P = P_z[z]

            # velocity
            v = v_z[z]

            ## calculate ##
            # mole fraction in the gas phase
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

            # mole fraction in the solid phase
            # MoFrsi_r0 = CosSpi_r/CosSp_r
            MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
                CosSpi_r)

            # TODO
            # dv/dz
            # gas velocity based on interstitial velocity [m/s]
            # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            # SuGaVe = InGaVe*BeVoFr
            # from ode eq. dv/dz
            SuGaVe = v

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = CoSp*SuGaVe*CrSeAr
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp*1000)
            GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # REVIEW
            # FIXME
            # viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaVi = np.zeros(compNo)  # f(T);
            # mixture viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaViMix = 2.5e-5  # f(yi,GaVi,MWs);
            # kinematic viscosity in the gas phase [m^2/s]
            GaKiViMix = GaViMix/GaDe

            # REVIEW
            # FIXME
            # add loop for each r point/constant
            # catalyst thermal conductivity [J/s.m.K]
            # CaThCo
            # membrane wall thermal conductivity [J/s.m.K]
            MeThCo = 1
            # thermal conductivity - gas phase [J/s.m.K]
            # GaThCoi = np.zeros(compNo)  # f(T);
            GaThCoi = np.array([0.278863993072407, 0.0353728593093126,	0.0378701882504170,
                               0.0397024608654616,	0.0412093811132403, 0.0457183034548015])
            # mixture thermal conductivity - gas phase [J/s.m.K]
            # convert
            GaThCoMix = 0.125
            # thermal conductivity - solid phase [J/s.m.K]
            # assume the same as gas phase
            # SoThCoi = np.zeros(compNo)  # f(T);
            SoThCoi = GaThCoi
            # mixture thermal conductivity - solid phase [J/s.m.K]
            SoThCoMix = 0.125
            # effective thermal conductivity - gas phase [J/s.m.K]
            # GaThCoEff = BeVoFr*GaThCoMix + (1 - BeVoFr)*CaThCo
            GaThCoEff = BeVoFr*GaThCoMix
            # effective thermal conductivity - solid phase [J/s.m.K]
            # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
            SoThCoEff = CaThCo*((1 - CaPo)/CaTo)

            # REVIEW
            # diffusivity coefficient - gas phase [m^2/s]
            # GaDii = np.zeros(compNo)  # gas_diffusivity_binary(yi,T,P0);
            GaDii = np.array([6.61512999110972e-06,	2.12995183554984e-06,	1.39108654241678e-06,
                             2.20809430865725e-06,	9.64429037148681e-07,	8.74374373632434e-07])
            # effective diffusivity - solid phase [m2/s]
            SoDiiEff = (CaPo/CaTo)*GaDii

            # REVIEW
            ### dimensionless numbers ###
            # Re Number
            ReNu = calReNoEq1(GaDe, SuGaVe, PaDi, GaViMix)
            # Sc Number
            ScNu = calScNoEq1(GaDe, GaViMix, GaDii)
            # Sh Number (choose method)
            ShNu = calShNoEq1(ScNu, ReNu, CONST_EQ_Sh['Frossling'])

            # REVIEW
            # mass transfer coefficient - gas/solid [m/s]
            MaTrCo = calMassTransferCoefficientEq1(ShNu, GaDii, PaDi)

            # NOTE
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
            for r in range(rNo):
                #
                # r0 = np.array(PackedBedReactorClass.modelReactions(
                #     P_z[z], Ts_r[r], MoFrsi_r[r], CaDe))

                # loop
                loopVars0 = (Ts_r[r], P_z[z], MoFrsi_r[r], CosSpi_r[r])

                # component formation rate [mol/m^3.s]
                # check unit
                r0 = np.array(reactionRateExe(
                    loopVars0, varisSet, ratesSet))

                # loop
                Ri_zr[z, r, :] = r0
                Ri_r[r, :] = r0

                # reset
                _riLoop = 0

                # REVIEW
                # component formation rate [kmol/m^3.s]
                # ri = np.zeros(compNo)
                # for k in range(compNo):
                #     # reset
                #     _riLoop = 0
                #     # number of reactions
                #     for m in range(len(reactionStochCoeff)):
                #         # number of components in each reaction
                #         for n in range(len(reactionStochCoeff[m])):
                #             # check component id
                #             if comList[k] == reactionStochCoeff[m][n][0]:
                #                 _riLoop += reactionStochCoeff[m][n][1] * \
                #                     Ri_r[r][m]
                #         ri_r0[r][k] = _riLoop

                ri_r[r] = componentFormationRate(
                    compNo, comList, reactionStochCoeff, Ri_r[r])

                # overall formation rate [kmol/m^3.s]
                OvR[r] = np.sum(ri_r[r])

            # NOTE
            ### enthalpy calculation ###
            # gas phase
            # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
            # Cp mean list
            GaCpMeanList = calMeanHeatCapacityAtConstantPressure(comList, T)
            # Cp mixture
            GaCpMeanMix = calMixtureHeatCapacityAtConstantPressure(
                MoFri, GaCpMeanList)
            # effective heat capacity - gas phase [kJ/kmol.K] | [J/mol.K]
            GaCpMeanMixEff = GaCpMeanMix*BeVoFr

            # FIXME
            # effective heat capacity - solid phase [kJ/m^3.K]
            SoCpMeanMixEff = CoSp*GaCpMeanMix*CaPo + (1-CaPo)*CaDe*CaSpHeCa

            # solid phase
            for r in range(rNo):
                # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
                # Cp mean list
                SoCpMeanList = calMeanHeatCapacityAtConstantPressure(
                    comList, Ts_r[r])
                # Cp mixture
                SoCpMeanMix[r] = calMixtureHeatCapacityAtConstantPressure(
                    MoFrsi_r[r], SoCpMeanList)

                # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
                # enthalpy change
                EnChList = np.array(
                    calEnthalpyChangeOfReaction(reactionListSorted, Ts_r[r]))
                # heat of reaction at T [kJ/kmol] | [J/mol]
                HeReT = np.array(EnChList + StHeRe25)
                # overall heat of reaction [kJ/m^3.s]
                # exothermic reaction (negative sign)
                # endothermic sign (positive sign)
                OvHeReT[r] = np.dot(Ri_r[r, :], HeReT)

            # REVIEW
            # Prandtl Number
            # MW kg/mol -> g/mol
            # MiMoWe_Conv = 1000*MiMoWe
            PrNu = calPrNoEq1(
                GaCpMeanMix, GaViMix, GaThCoMix, MiMoWe)
            # Nu number
            NuNu = calNuNoEq1(PrNu, ReNu)
            # heat transfer coefficient - gas/solid [J/m^2.s.K]
            HeTrCo = calHeatTransferCoefficientEq1(NuNu, GaThCoMix, PaDi)

            # REVIEW
            # heat transfer coefficient - medium side [J/m2.s.K]
            # hs = heat_transfer_coefficient_shell(T,Tv,Pv,Pa);
            # overall heat transfer coefficient [J/m2.s.K]
            # U = overall_heat_transfer_coefficient(hfs,kwall,do,di,L);
            # heat transfer coefficient - permeate side [J/m2.s.K]

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m^2/m^3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            Ua = U*a
            # external heat [kJ/m^3.s]
            # if Tm == 0:
            #     # adiabatic
            #     Qm0 = 0
            # else:
            #     # heat added/removed from the reactor
            #     # Tm > T: heat is added (positive sign)
            #     # T > Tm: heat removed (negative sign)
            #     Qm0 = (Ua*(Tm - T))*1e-3

            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T, U, a, 'kJ/m^3.s')

            # NOTE
            # mass transfer between
            for i in range(compNo):
                ### gas phase ###
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # inward flux [kmol/m^2.s]
                MoFli_z[i] = MaTrCo[i]*(Ci_c - CosSpi_cat[i])

            # total mass transfer between gas and solid phases [kmol/m^3]
            ToMaTrBeGaSo_z = np.sum(MoFli_z)*SpSuAr

            # NOTE
            # velocity from global concentration
            # check BC
            # if z == 0:
            #     # BC1
            #     T_b = T0
            # else:
            #     # interior nodes
            #     T_b = T_z[z - 1]

            # check BC
            if z == 0:
                # BC1
                constT_BC1 = (GaThCoEff)/(MoFl*GaCpMeanMix/1000)
                # next node
                T_f = T_z[z+1]
                # previous node
                T_b = (T0*dz + constT_BC1*T_f)/(dz + constT_BC1)
            elif z == zNo - 1:
                # BC2
                # previous node
                T_b = T_z[z - 1]
                # next node
                T_f = 0
            else:
                # interior nodes
                T_b = T_z[z-1]
                # next node
                T_f = T_z[z+1]

            dxdt_v_T = (T_z[z] - T_b)/dz
            # CoSp x 1000
            # OvR x 1000
            dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
                                      ((1/T)*dxdt_P - (P/T**2)*dxdt_v_T) - ToMaTrBeGaSo_z*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            v_z[z+1] = dxdt_v*dz + v_z[z]

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # loop vars
            const_F1 = 1/BeVoFr
            # [kmol/m^2.s][kJ/kmol.K]=[kJ/m^2.s.K]
            const_T1 = MoFl*GaCpMeanMix
            # [kmol/m^3][kJ/kmol.K]=[kJ/m^3.K]
            const_T2 = 1/(CoSp*GaCpMeanMixEff)

            # catalyst
            const_Cs1 = 1/(CaPo*(PaRa**2))
            const_Ts1 = 1/(SoCpMeanMixEff*(PaRa**2))

            # bulk temperature [K]
            T_c = T_z[z]

            # REVIEW
            # gas-solid interface BC
            # concentration [m/s]*[m^2/s]=[1/m]
            betaC = PaRa*(MaTrCo/SoDiiEff)
            # temperature
            betaT = -1*((HeTrCo*PaRa)/SoThCoEff)

            # universal index [j,i]
            # UISet = z*(rNo + 1)

            # NOTE
            # concentration [mol/m^3]
            for i in range(compNo):

                ### gas phase ###
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # check BC
                if z == 0:
                    # BC1
                    constC_BC1 = GaDii[i]*BeVoFr/v_z[z]
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_b = (1/(constC_BC1 + dz)) * \
                        (SpCoi0[i]*dz + constC_BC1*(Ci_f))
                elif z == zNo - 1:
                    # BC2
                    # forward difference
                    Ci_f = 0
                    # previous node
                    Ci_b = max(SpCoi_z[i][z - 1], CONST.EPS_CONST)
                else:
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    # interior nodes
                    Ci_b = max(SpCoi_z[i][z - 1], CONST.EPS_CONST)

                # cal differentiate
                # backward difference
                dCdz = (Ci_c - Ci_b)/dz
                # central difference for dispersion
                d2Cdz2 = (Ci_b - 2*Ci_c + Ci_f)/(dz**2)
                # dispersion term [kmol/m^3.s]
                _dispersionFluxC = GaDii[i]*BeVoFr*d2Cdz2
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # inward flux [kmol/m^2.s]
                # MoFli_z[i] = MaTrCo[i]*(Ci_c - CosSpi_cat[i])
                # mass balance
                # convective, dispersion, inward flux
                dxdt_F = const_F1 * \
                    (-v_z[z]*dCdz - Ci_c*dxdt_v +
                     _dispersionFluxC - MoFli_z[i]*SpSuAr)
                dxdtMat[i][0][z] = dxdt_F

                ### solid phase ###
                # bulk concentration [kmol/m^3]
                # Ci_c
                # bulk temperature [K]
                # T_c
                # species concentration at different points of particle radius [rNo]
                # [Cs[3], Cs[2], Cs[1], Cs[0]]
                _Cs_r = CosSpi_r[:, i].flatten()

                # updated concentration gas-solid interface
                # shape(rNo,1)
                _Cs_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                    _Cs_r, Ci_c, betaC[i])

                # dC/dt list
                dCsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                    _Cs_r_Updated, SoDiiEff[i], (PaRa**2)*ri_r[:, i])

                for r in range(rNo):
                    # update
                    dxdtMat[i][r+1][z] = const_Cs1*dCsdti[r]

            # NOTE
            # energy balance (temperature) [K]
            # temp [K]
            # T_c = T_z[z]

            # temperature at different points of particle radius [rNo]
            # Ts[3], Ts[2], Ts[1], Ts[0]
            _Ts_r = Ts_r.flatten()

            # check BC
            if z == 0:
                # BC1
                constT_BC1 = (GaThCoEff)/(MoFl*GaCpMeanMix*1000)
                # next node
                T_f = T_z[z+1]
                # previous node
                T_b = (T0*dz + constT_BC1*T_f)/(dz + constT_BC1)
            elif z == zNo - 1:
                # BC2
                # previous node
                T_b = T_z[z - 1]
                # next node
                T_f = 0
            else:
                # interior nodes
                T_b = T_z[z - 1]
                # next node
                T_f = T_z[z+1]

            # cal differentiate
            # backward difference
            dTdz = (T_c - T_b)/dz
            # central difference
            d2Tdz2 = (T_b - 2*T_c + T_f)/(dz**2)
            # FIXME
            # dispersion flux [kJ/m^3.s]
            _dispersionFluxT = (GaThCoEff*d2Tdz2)*1e-3*0
            # temperature in the catalyst surface [K]
            # Ts_cat
            # outward flux [kJ/m^2.s]
            InFlT = HeTrCo*(_Ts_r[0] - T_c)*1e-3
            # total heat transfer between gas and solid [kJ/m^3.s]
            ToHeTrBeGaSo_z = InFlT*SpSuAr
            # convective flux, diffusive flux, enthalpy of reaction, cooling heat
            dxdt_T = const_T2 * \
                (-const_T1*dTdz + _dispersionFluxT + ToHeTrBeGaSo_z + Qm)
            dxdtMat[indexT][0][z] = dxdt_T

            ### solid phase ###
            # _Ts_r
            # T[n], T[n-1], ..., T[0]
            # updated temperature gas--solid interface
            _Ts_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                _Ts_r, T_c, betaT)

            # dC/dt list
            # convert
            # [J/s.m.K] => [kJ/s.m.K]
            SoThCoEff_Conv = SoThCoEff/1000
            # OvHeReT [kJ/m^3.s]
            OvHeReT_Conv = -1*OvHeReT
            dTsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                _Ts_r_Updated, SoThCoEff_Conv, (PaRa**2)*OvHeReT_Conv)

            for r in range(rNo):
                # update
                dxdtMat[indexT][r+1][z] = const_Ts1*dTsdti[r]

        # NOTE
        # set time
        # flat
        dxdt = dxdtMat.flatten().tolist()

        return dxdt

# NOTE
# dynamic heterogenous modeling

    def runM7(self):
        """
        M7 modeling case (dimensionless)
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static), Cci, Tc (dynamic, for catalyst)
            CT, GaDe = f(P, T, n)
        numerical method: finite difference
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']
        solverMesh = solverConfig['mesh']
        solverMeshSet = True if solverMesh == "normal" else False

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']
        # numerical method
        numericalMethod = self.modelInput['operating-conditions']['numerical-method']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        # labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # diffusivity coefficient - gas phase [m^2/s]
        GaDii0 = self.modelInput['feed']['diffusivity']
        # gas viscosity [Pa.s]
        GaVii0 = self.modelInput['feed']['viscosity']
        # gas mixture viscosity [Pa.s]
        GaViMix0 = self.modelInput['feed']['mixture-viscosity']

        # thermal conductivity - gas phase [J/s.m.K]
        GaThCoi0 = self.modelInput['feed']['thermal-conductivity']
        # mixture thermal conductivity - gas phase [J/s.m.K]
        GaThCoMix0 = self.modelInput['feed']['mixture-thermal-conductivity']

        # REVIEW
        # domain length
        DoLe = 1
        # orthogonal collocation points in the r direction
        # rNo = solverSetting['S2']['rNo']
        if numericalMethod == "fdm":
            # finite difference points in the r direction
            rNo = solverSetting['T1']['rNo']['fdm']
        elif numericalMethod == "oc":
            # orthogonal collocation points in the r direction
            rNo = solverSetting['T1']['rNo']['oc']
        else:
            raise

        # mesh setting
        zMesh = solverSetting['T1']['zMesh']
        # number of nodes
        zNoNo = zMesh['zNoNo']
        # domain length section
        DoLeSe = zMesh['DoLeSe']
        # mesh refinement degree
        MeReDe = zMesh['MeReDe']
        # mesh installment
        if solverMeshSet is False:
            zMeshRes = FiDiMeshGenerator(zNoNo, DoLe, DoLeSe, MeReDe)
            # finite difference points
            dataXs = zMeshRes['data1']
            # dz lengths
            dzs = zMeshRes['data2']
            # finite difference point number
            zNo = zMeshRes['data3']
            # R ratio
            zR = zMeshRes['data4']
            # dz
            dz = zMeshRes['data5']
        else:
            # finite difference points in the z direction
            zNo = solverSetting['T1']['zNo']
            # length list [reactor length]
            dataXs = np.linspace(0, DoLe, zNo)
            # element size - dz [m]
            dz = DoLe/(zNo-1)
            # reset
            dzs = []
            zR = []

        ### calculation ###
        # mole fraction in the gas phase
        MoFri0 = np.array(rmtUtil.moleFractionFromConcentrationSpecies(SpCoi0))

        # mixture molecular weight [kg/mol]
        MiMoWe0 = rmtUtil.mixtureMolecularWeight(MoFri0, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe0 = calDensityIG(MiMoWe0, SpCo0*1000)

        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
        # Cp mean list
        GaCpMeanList0 = calMeanHeatCapacityAtConstantPressure(compList, T)
        # Cp mixture
        GaCpMeanMix0 = calMixtureHeatCapacityAtConstantPressure(
            MoFri0, GaCpMeanList0)

        # thermal diffusivity in the gas phase [m^2/s]
        GaThDi = calThermalDiffusivity(
            GaThCoMix0, GaDe0, GaCpMeanMix0, MiMoWe0)

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total number
        varNoConInSolid = varNoConInSolidBlock*zNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of layers
        noLayer = compNo + 1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = rNo + 1
        # number of var columns [i]
        varNoColumns = zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoRows, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayer - 1):
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # gas phase
                        if i == 0:
                            IV2D[m][j][i] = SpCoi0[m]/np.max(SpCoi0)
                        else:
                            IV2D[m][j][i] = SpCoi0[m]/np.max(SpCoi0)
                    else:
                        # solid phase
                        # SpCoi0[m]/np.max(SpCoi0)  # SpCoi0[m]
                        IV2D[m][j][i] = 1e-6

        # temperature
        for i in range(varNoColumns):
            for j in range(varNoRows):
                # separate phase
                if j == 0:
                    # gas phase
                    if i == 0:
                        IV2D[noLayer - 1][j][i] = 0  # T
                    else:
                        IV2D[noLayer - 1][j][i] = 0  # T
                else:
                    # solid phase
                    IV2D[noLayer - 1][j][i] = 0  # T

        # flatten IV
        IV = IV2D.flatten()

        # print(f"IV: {IV}")

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted
        # reaction coefficient
        reactionStochCoeff = self.reactionStochCoeffList

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # REVIEW
        # solver setting

        # NOTE
        ### dimensionless analysis ###
        # concentration [kmol/m^3]
        Cif = np.copy(SpCoi0)
        # total concentration
        Cf = SpCo0
        # temperature [K]
        Tf = T
        # superficial velocity [m/s]
        vf = SuGaVe0
        # length [m]
        zf = ReLe
        # diffusivity [m^2/s]
        Dif = np.copy(GaDii0)
        # heat capacity at constant pressure [J/mol.K] | [kJ/kmol.K]
        Cpif = np.copy(GaCpMeanList0)
        # mixture heat capacity [J/mol.K] | [kJ/kmol.K]
        Cpf = GaCpMeanMix0
        # radius
        rf = PaDi/2

        # gas phase
        # mass convective term - (list) [kmol/m^3.s]
        _Cif = Cif if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.repeat(
            np.max(Cif), compNo)
        GaMaCoTe0 = (vf/zf)*_Cif
        # mass diffusive term - (list)  [kmol/m^3.s]
        GaMaDiTe0 = (1/zf**2)*(_Cif*Dif)
        # heat convective term [kJ/m^3.s]
        GaHeCoTe0 = (GaDe0*vf*Tf*(Cpf/MiMoWe0)/zf)*1e-3
        # heat diffusive term [kJ/m^3.s]
        GaHeDiTe0 = (Tf*GaThCoMix0/zf**2)*1e-3

        # solid phase
        # mass diffusive term - (list)  [kmol/m^3.s]
        SoMaDiTe0 = (Dif*_Cif)/rf**2
        # heat diffusive term [kJ/m^3.s]
        SoHeDiTe0 = (GaThCoMix0*Tf/rf**2)*1e-3

        ### dimensionless numbers ###
        # Re Number
        ReNu0 = calReNoEq1(GaDe0, SuGaVe0, PaDi, GaViMix0)
        # Sc Number
        ScNu0 = calScNoEq1(GaDe0, GaViMix0, GaDii0)
        # Sh Number (choose method)
        ShNu0 = calShNoEq1(ScNu0, ReNu0, CONST_EQ_Sh['Frossling'])
        # Prandtl Number
        PrNu0 = calPrNoEq1(GaCpMeanMix0, GaViMix0, GaThCoMix0, MiMoWe0)
        # Nu number
        NuNu0 = calNuNoEq1(PrNu0, ReNu0)
        # Strouhal number
        StNu = 1
        # Peclet number - mass transfer
        PeNuMa0 = (vf*zf)/Dif
        # Peclet number - heat transfer
        PeNuHe0 = (zf*GaDe0*(Cpf/MiMoWe0)*vf)/GaThCoMix0

        ### transfer coefficient ###
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = calMassTransferCoefficientEq1(ShNu0, GaDii0, PaDi)
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = calHeatTransferCoefficientEq1(NuNu0, GaThCoMix0, PaDi)

        # fun parameters
        FunParam = {
            "compList": compList,
            "const": {
                "CrSeAr": CrSeAr,
                "MoWei": MoWei,
                "StHeRe25": StHeRe25,
                "GaMiVi": GaViMix0,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T,
                "SuGaVe0": SuGaVe0,
                "GaDii0": GaDii0,
                "GaThCoi0": GaThCoi0,
                "GaVii0": GaVii0,
                "GaDe0": GaDe0,
                "GaCpMeanMix0": GaCpMeanMix0,
                "GaThCoMix0": GaThCoMix0
            },
            "meshSetting": {
                "solverMesh": solverMesh,
                "solverMeshSet": solverMeshSet,
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz,
                "dzs": dzs,
                "zR": zR,
                "zNoNo": zNoNo
            },
            "solverSetting": {
                "dFdz": solverSetting['T1']['dFdz'],
                "d2Fdz2": solverSetting['T1']['d2Fdz2'],
                "dTdz": solverSetting['T1']['dTdz'],
                "d2Tdz2": solverSetting['T1']['d2Tdz2'],
            },
            "reactionRateExpr": reactionRateExpr

        }

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "zf": zf,
            "Dif": Dif,
            "Cpif": Cpif,
            "Cpf": Cpf,
            "rf": rf,
            "GaMaCoTe0": GaMaCoTe0,
            "GaMaDiTe0": GaMaDiTe0,
            "GaHeCoTe0": GaHeCoTe0,
            "GaHeDiTe0": GaHeDiTe0,
            "ReNu0": ReNu0,
            "ScNu0": ScNu0,
            "ShNu0": ShNu0,
            "PrNu0": PrNu0,
            "PeNuMa0": PeNuMa0,
            "PeNuHe0": PeNuHe0,
            "MaTrCo": MaTrCo,
            "HeTrCo": HeTrCo,
            "SoMaDiTe0": SoMaDiTe0,
            "SoHeDiTe0": SoHeDiTe0
        }

        # time span
        tNo = solverSetting['T1']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['T1']['timesNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((varNo, tNo, zNo))
        #

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # FIXME
        n = solverSetting['T1']['ode-solver']['PreCorr3']['n']
        # t0 = 0
        # tn = 5
        # t = np.linspace(t0, tn, n+1)
        paramsSet = (reactionListSorted, reactionStochCoeff,
                     FunParam, DimensionlessAnalysisParams)
        funSet = PackedBedReactorClass.modelEquationM7

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)
            print(f"time: {t} seconds")

            # ode call
            if solverIVP == "AM":
                # sol = AdBash3(t[0], t[1], n, IV, funSet, paramsSet)
                # PreCorr3
                sol = PreCorr3(t[0], t[1], n, IV, funSet, paramsSet)
                successStatus = True
                # time interval
                dataTime = t
                # all results
                # components, temperature layers
                dataYs = sol
            else:
                # method [1]: LSODA, [2]: BDF, [3]: Radau
                # options
                solverOptions = {
                    "atol": 1e-7
                }
                sol = solve_ivp(funSet, t, IV, method=solverIVP,
                                t_eval=times,  args=(paramsSet,))
                # ode result
                successStatus = sol.success
                # check
                if successStatus is False:
                    raise
                # time interval
                dataTime = sol.t
                # all results
                # components, temperature layers
                dataYs = sol.y

            # REVIEW
            # post-processing result
            # std format
            dataYs_Reshaped = np.reshape(
                dataYs[:, -1], (noLayer, varNoRows, varNoColumns))

            # component concentration [kmol/m^3]
            # Ci and Cs
            # dataYs1 = dataYs[0:varNoCon, -1]
            # 3d matrix
            # dataYs1_Reshaped = np.reshape(
            #     dataYs1, (compNo, varNoRows, varNoColumns))

            dataYs1_Reshaped = dataYs_Reshaped[:-1]

            # gas phase
            dataYs1GasPhase = dataYs1_Reshaped[:, 0, :]
            # solid phase
            dataYs1SolidPhase = dataYs1_Reshaped[:, 1:, :]

            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1GasPhase, axis=0)
            dataYs1_MoFri = dataYs1GasPhase/dataYs1_Ctot

            # temperature - 2d matrix
            dataYs2_Reshaped = dataYs_Reshaped[indexTemp]
            # gas phase
            dataYs2GasPhase = dataYs2_Reshaped[0, :].reshape((1, zNo))
            # solid phase
            dataYs2SolidPhase = dataYs2_Reshaped[1:, :]

            # combine
            _dataYs = np.concatenate(
                (dataYs1_MoFri, dataYs2GasPhase), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCon": dataYs1GasPhase,
                "dataYTemp": dataYs2GasPhase,
                "dataYs": _dataYs,
                "dataYCons": dataYs1SolidPhase,
                "dataYTemps": dataYs2SolidPhase,
            })

            for m in range(varNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # NOTE
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        # ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        # ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        # ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        # ssdataLists = [ssdataList[0:compNo],
        #                ssdataList[indexTemp]]
        # subplot result
        # pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        # plot info
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # display result at specific time
        # subplot result
        xLabelSet = "Dimensionless Reactor Length"
        yLabelSet = "Dimensionless Concentration"

        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataYs']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(dataLists, xLabelSet, yLabelSet, plotTitle)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _name = labelList[i] + " at t=" + str(opTSpan[t+1])

                _labelNameTime.append(_name)

            dataList = pltc.plots2DSetDataList(XYList, _labelNameTime)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select items
        # indices = [0, 2, -1]
        # selected_elements = [_dataListsLoop[index] for index in indices]
        # select datalist
        _dataListsSelected = selectFromListByIndex([1, -1], _dataListsLoop)

        # subplot result
        # pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM7(t, y, paramsSet):
        """
            M7 model [dynamic modeling]
            mass, energy, and momentum balance equations
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
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]
                    meshSetting:
                        solverMesh: mesh installment
                        solverMeshSet: 
                            true: normal
                            false: mesh refinement
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                        dzs: differential length list [-]
                        zR: z ratio
                        zNoNo: number of nodes in the dense and normal sections
                    solverSetting:
                    reactionRateExpr: reaction rate expressions
                DimensionlessAnalysisParams:
                    Cif: feed concentration [kmol/m^3]
                    Tf: feed temperature
                    vf: feed superficial velocity [m/s]
                    zf: domain length [m]
                    Dif: diffusivity coefficient of component [m^2/s]
                    Cpif: feed heat capacity at constat pressure [kJ/kmol.K] | [J/mol.K]
                    rf: particle radius [m]
                    GaMaCoTe0: feed mass convective term of gas phase [kmol/m^3.s]
                    GaMaDiTe0: feed mass diffusive term of gas phase [kmol/m^3.s]
                    GaHeCoTe0: feed heat convective term of gas phase [kJ/m^3.s]
                    GaHeDiTe0, feed heat diffusive term of gas phase [kJ/m^3.s]
                    SoMaDiTe0: feed mass diffusive term of solid phase [kmol/m^3.s]
                    SoHeDiTe0: feed heat diffusive term of solid phase [kJ/m^3.s]
                    ReNu0: Reynolds number
                    ScNu0: Schmidt number
                    ShNu0: Sherwood number
                    PrNu0: Prandtl number
                    PeNuMa0: mass Peclet number
                    PeNuHe0: heat Peclet number 
                    MaTrCo: mass transfer coefficient - gas/solid [m/s]
                    HeTrCo: heat transfer coefficient - gas/solid [J/m^2.s.K]
        """
        # params
        reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams = paramsSet
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

        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']
        # catalyst porosity
        CaPo = ReSpec['CaPo']
        # catalyst tortuosity
        CaTo = ReSpec['CaTo']
        # catalyst thermal conductivity [J/K.m.s]
        CaThCo = ReSpec['CaThCo']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']
        # inlet superficial velocity [m/s]
        # SuGaVe0 = constBC1['SuGaVe0']
        # inlet diffusivity coefficient [m^2]
        GaDii0 = constBC1['GaDii0']
        # inlet gas thermal conductivity [J/s.m.K]
        GaThCoi0 = constBC1['GaThCoi0']
        # gas viscosity
        GaVii0 = constBC1['GaVii0']
        # gas density [kg/m^3]
        GaDe0 = constBC1['GaDe0']
        # heat capacity at constant pressure [kJ/kmol.K] | [J/mol.K]
        GaCpMeanMix0 = constBC1['GaCpMeanMix0']
        # gas thermal conductivity [J/s.m.K]
        GaThCoMix0 = constBC1['GaThCoMix0']

        # mesh setting
        meshSetting = FunParam['meshSetting']
        # mesh installment
        solverMesh = meshSetting['solverMesh']
        # mesh refinement
        solverMeshSet = meshSetting['solverMeshSet']
        # number of layers
        noLayer = meshSetting['noLayer']
        # var no in each layer
        varNoLayer = meshSetting['varNoLayer']
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = meshSetting['varNoLayerT']
        # number of var rows [j]
        varNoRows = meshSetting['varNoRows']
        # number of var columns [i]
        varNoColumns = meshSetting['varNoColumns']
        # rNo
        rNo = meshSetting['rNo']
        # zNo
        zNo = meshSetting['zNo']
        # dz [m]
        dz = meshSetting['dz']
        # dzs [m]/[-]
        dzs = meshSetting['dzs']
        # R ratio
        zR = meshSetting['zR']
        # number of nodes in the dense and normal sections
        zNoNo = meshSetting['zNoNo']
        # dense
        zNoNoDense = zNoNo[0]
        # normal
        zNoNoNormal = zNoNo[1]

        # solver setting
        solverSetting = FunParam['solverSetting']
        # mass balance equation
        DIFF1_C_SET = solverSetting['dFdz']
        DIFF2_C_SET_BC1 = solverSetting['d2Fdz2']['BC1']
        DIFF2_C_SET_BC2 = solverSetting['d2Fdz2']['BC2']
        DIFF2_C_SET_G = solverSetting['d2Fdz2']['G']
        # energy balance equation
        DIFF1_T_SET = solverSetting['dTdz']
        DIFF2_T_SET_BC1 = solverSetting['d2Tdz2']['BC1']
        DIFF2_T_SET_BC2 = solverSetting['d2Tdz2']['BC2']
        DIFF2_T_SET_G = solverSetting['d2Tdz2']['G']

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # dimensionless analysis params

        #  feed concentration [kmol/m^3]
        Cif = DimensionlessAnalysisParams['Cif']
        # feed temperature
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # domain length [m]
        zf = DimensionlessAnalysisParams['zf']
        # particle radius [m]
        rf = DimensionlessAnalysisParams['rf']
        # diffusivity coefficient of component [m^2/s]
        Dif = DimensionlessAnalysisParams['Dif']
        # feed heat capacity at constat pressure
        Cpif = DimensionlessAnalysisParams['Cpif']
        # feed mass convective term of gas phase [kmol/m^3.s]
        GaMaCoTe0 = DimensionlessAnalysisParams['GaMaCoTe0']
        # feed mass diffusive term of gas phase [kmol/m^3.s]
        GaMaDiTe0 = DimensionlessAnalysisParams['GaMaDiTe0']
        # feed heat convective term of gas phase [kJ/m^3.s]
        GaHeCoTe0 = DimensionlessAnalysisParams['GaHeCoTe0']
        # feed heat diffusive term of gas phase [kJ/m^3.s]
        GaHeDiTe0 = DimensionlessAnalysisParams['GaHeDiTe0']
        # feed mass diffusive term of solid phase [kmol/m^3.s]
        SoMaDiTe0 = DimensionlessAnalysisParams['SoMaDiTe0']
        # feed heat diffusive term of solid phase [kJ/m^3.s]
        SoHeDiTe0 = DimensionlessAnalysisParams['SoHeDiTe0']
        # Reynolds number
        ReNu = DimensionlessAnalysisParams['ReNu0']
        # Schmidt number
        ScNu = DimensionlessAnalysisParams['ScNu0']
        # Sherwood number
        ShNu = DimensionlessAnalysisParams['ShNu0']
        # Prandtl number
        PrNu = DimensionlessAnalysisParams['PrNu0']
        # mass Peclet number
        PeNuMa0 = DimensionlessAnalysisParams['PeNuMa0']
        # heat Peclet number
        PeNuHe0 = DimensionlessAnalysisParams['PeNuHe0']
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = DimensionlessAnalysisParams['MaTrCo']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = DimensionlessAnalysisParams['HeTrCo']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexV = indexP + 1

        # calculate
        # particle radius
        PaRa = PaDi/2
        # specific surface area exposed to the free fluid [m^2/m^3]
        SpSuAr = (3/PaRa)*(1 - BeVoFr)

        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # interstitial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate in the solid phase
        Ri_z = np.zeros((zNo, reactionListNo))
        Ri_zr = np.zeros((zNo, rNo, reactionListNo))
        Ri_r = np.zeros((rNo, reactionListNo))
        # reaction rate
        # ri = np.zeros(compNo) # deprecate
        # ri0 = np.zeros(compNo) # deprecate
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        # overall reaction
        OvR = np.zeros(rNo)
        # overall enthalpy
        OvHeReT = np.zeros(rNo)
        # heat capacity at constant pressure
        SoCpMeanMix = np.zeros(rNo)
        # effective heat capacity at constant pressure
        SoCpMeanMixEff = np.zeros(rNo)
        # dimensionless analysis
        SoCpMeanMixEff_ReVa = np.zeros(rNo)

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # superficial gas velocity [m/s]
        v_z = np.zeros(zNo + 1)
        v_z[0] = SuGaVe0

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (noLayer, varNoRows, varNoColumns))

        # all species concentration in gas & solid phase
        SpCo_mz = np.zeros((noLayer - 1, varNoRows, varNoColumns))
        # all species concentration in gas phase [kmol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        # all species concentration in solid phase (catalyst) [kmol/m^3]
        SpCosi_mzr = np.zeros((compNo, rNo, zNo))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCo_mz[m] = _SpCoi
        # concentration in the gas phase [kmol/m^3]
        for m in range(compNo):
            for j in range(varNoRows):
                if j == 0:
                    # gas phase
                    SpCoi_z[m, :] = SpCo_mz[m, j, :]
                else:
                    # solid phase
                    SpCosi_mzr[m, j-1, :] = SpCo_mz[m, j, :]

        # species concentration in gas phase [kmol/m^3]
        CoSpi = np.zeros(compNo)
        # dimensionless analysis
        CoSpi_ReVa = np.zeros(compNo)
        # total concentration [kmol/m^3]
        CoSp = 0
        # species concentration in solid phase (catalyst) [kmol/m^3]
        # shape
        CosSpiMatShape = (rNo, compNo)
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)
        # total concentration in the solid phase [kmol/m^3]
        CosSp_r = np.zeros(rNo)

        # flux
        MoFli_z = np.zeros(compNo)

        # NOTE
        # temperature [K]
        T_mz = np.zeros((varNoRows, varNoColumns))
        T_mz = yLoop[noLayer - 1]
        # temperature in the gas phase
        T_z = np.zeros(zNo)
        T_z = T_mz[0, :]
        # temperature in solid phase
        Ts_z = np.zeros((rNo, zNo))
        Ts_z = T_mz[1:]
        # temperature in the solid phase
        Ts_r = np.zeros(rNo)

        # diff/dt
        # dxdt = []
        # matrix
        # dxdtMat = np.zeros((varNo, zNo))
        dxdtMat = np.zeros((noLayer, varNoRows, varNoColumns))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(varNoColumns):
            ## block ##

            # concentration species in the gas phase [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = max(_SpCoi_z, CONST.EPS_CONST)
                # REVIEW
                # dimensionless analysis: real value
                SpCoi0_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                CoSpi_ReVa[i] = rmtUtil.calRealDiLessValue(
                    CoSpi[i], SpCoi0_Set)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)
            # dimensionless analysis: real value
            CoSp_ReVa = np.sum(CoSpi_ReVa)

            # FIXME
            # concentration species in the solid phase [kmol/m^3]
            # display concentration list in each oc point (rNo)
            for i in range(compNo):
                for r in range(rNo):
                    _CosSpi_z = SpCosi_mzr[i][r][z]
                    CosSpi_r[r][i] = max(_CosSpi_z, CONST.EPS_CONST)
                    # REVIEW
                    # dimensionless analysis: real value
                    SpCoi0_r_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                        SpCoi0)
                    CosSpi_r_ReVa[r][i] = rmtUtil.calRealDiLessValue(
                        CosSpi_r[r][i], SpCoi0_r_Set)

            # total concentration in the solid phase [kmol/m^3]
            CosSp_r = np.sum(CosSpi_r, axis=1).reshape((rNo, 1))
            # dimensionless analysis: real value
            CosSp_r_ReVa = np.sum(CosSpi_r_ReVa, axis=1).reshape((rNo, 1))

            # concentration in the outer surface of the catalyst [kmol/m^3]
            CosSpi_cat = CosSpi_r[0]
            # dimensionless analysis
            CosSpi_cat_DiLeVa = CosSpi_r[0, :]

            # temperature [K]
            T = T_z[z]
            T_ReVa = rmtUtil.calRealDiLessValue(T, T0, "TEMP")
            # temperature in the solid phase (for each point)
            # Ts[3], Ts[2], Ts[1], Ts[0]
            Ts_r = Ts_z[:, z]
            Ts_r_ReVa0 = rmtUtil.calRealDiLessValue(Ts_r, Tf, "TEMP")
            Ts_r_ReVa = np.reshape(Ts_r_ReVa0, -1)

            # pressure [Pa]
            P = P_z[z]

            # FIXME
            # velocity
            # dimensionless value
            # v = v_z[z]
            v = 1

            ## calculate ##
            # mole fraction in the gas phase
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi_ReVa))

            # mole fraction in the solid phase
            # MoFrsi_r0 = CosSpi_r/CosSp_r
            MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
                CosSpi_r_ReVa)

            # TODO
            # dv/dz
            # gas velocity based on interstitial velocity [m/s]
            # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            # SuGaVe = InGaVe*BeVoFr
            # from ode eq. dv/dz
            SuGaVe = v
            # dimensionless analysis
            SuGaVe_ReVa = rmtUtil.calRealDiLessValue(SuGaVe, SuGaVe0)

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = calMolarFlowRate(CoSp_ReVa, SuGaVe_ReVa, CrSeAr)
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp_ReVa*1000)
            # GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)
            # dimensionless value
            GaDe_DiLeVa = rmtUtil.calDiLessValue(GaDe, GaDe0)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe_ReVa/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe_ReVa**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # REVIEW
            # FIXME
            # viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaVii = GaVii0 if MODEL_SETTING['GaVii'] == "FIX" else calTest()
            # mixture viscosity in the gas phase [Pa.s] | [kg/m.s]
            # FIXME
            GaViMix = 2.5e-5  # f(yi,GaVi,MWs);
            # kinematic viscosity in the gas phase [m^2/s]
            GaKiViMix = GaViMix/GaDe

            # REVIEW
            # FIXME
            # solid gas thermal conductivity
            SoThCoMix0 = GaThCoMix0
            # add loop for each r point/constant
            # catalyst thermal conductivity [J/s.m.K]
            # CaThCo
            # membrane wall thermal conductivity [J/s.m.K]
            MeThCo = 1
            # thermal conductivity - gas phase [J/s.m.K]
            # GaThCoi = np.zeros(compNo)  # f(T);
            GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest(
            )
            # dimensionless
            GaThCoi_DiLe = GaThCoi/GaThCoi0
            # FIXME
            # mixture thermal conductivity - gas phase [J/s.m.K]
            GaThCoMix = GaThCoMix0
            # dimensionless analysis
            GaThCoMix_DiLeVa = GaThCoMix/GaThCoMix0
            # thermal conductivity - solid phase [J/s.m.K]
            # assume the same as gas phase
            # SoThCoi = np.zeros(compNo)  # f(T);
            SoThCoi = GaThCoi
            # mixture thermal conductivity - solid phase [J/s.m.K]
            SoThCoMix = GaThCoMix0
            # dimensionless analysis
            SoThCoMix_DiLeVa = SoThCoMix/SoThCoMix0
            # effective thermal conductivity - gas phase [J/s.m.K]
            # GaThCoEff = BeVoFr*GaThCoMix + (1 - BeVoFr)*CaThCo
            GaThCoEff = BeVoFr*GaThCoMix
            # dimensionless analysis
            GaThCoEff_DiLeVa = BeVoFr*GaThCoMix_DiLeVa
            # FIXME
            # effective thermal conductivity - solid phase [J/s.m.K]
            # assume identical to gas phase
            # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
            # SoThCoEff = CaThCo*((1 - CaPo)/CaTo)
            SoThCoEff = CaPo*SoThCoMix
            # dimensionless analysis
            # SoThCoEff_DiLeVa = GaThCoMix_DiLeVa*((1 - CaPo)/CaTo)
            SoThCoEff_DiLeVa = CaPo*SoThCoMix_DiLeVa

            # REVIEW
            # diffusivity coefficient - gas phase [m^2/s]
            GaDii = GaDii0 if MODEL_SETTING['GaDii'] == "FIX" else calTest()
            # dimensionless analysis
            GaDii_DiLeVa = GaDii/GaDii0
            # effective diffusivity coefficient - gas phase
            GaDiiEff = GaDii*BeVoFr
            # dimensionless analysis
            GaDiiEff_DiLeVa = GaDiiEff/GaDii0
            # effective diffusivity - solid phase [m^2/s]
            SoDiiEff = (CaPo/CaTo)*GaDii
            # dimensionless analysis
            SoDiiEff_DiLe = (CaPo/CaTo)*GaDii_DiLeVa

            # REVIEW
            if MODEL_SETTING['MaTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Re Number
                ReNu = calReNoEq1(GaDe, SuGaVe, PaDi, GaViMix)
                # Sc Number
                ScNu = calScNoEq1(GaDe, GaViMix, GaDii)
                # Sh Number (choose method)
                ShNu = calShNoEq1(ScNu, ReNu, CONST_EQ_Sh['Frossling'])

                # mass transfer coefficient - gas/solid [m/s]
                MaTrCo = calMassTransferCoefficientEq1(ShNu, GaDii, PaDi)

            # NOTE
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
            for r in range(rNo):
                # loop
                loopVars0 = (Ts_r_ReVa[r], P_z[z],
                             MoFrsi_r[r], CosSpi_r_ReVa[r])

                # component formation rate [mol/m^3.s]
                # check unit
                r0 = np.array(reactionRateExe(
                    loopVars0, varisSet, ratesSet))

                # loop
                Ri_zr[z, r, :] = r0
                Ri_r[r, :] = r0

                # component formation rate [kmol/m^3.s]
                ri_r[r] = componentFormationRate(
                    compNo, comList, reactionStochCoeff, Ri_r[r])

                # overall formation rate [kmol/m^3.s]
                OvR[r] = np.sum(ri_r[r])

            # NOTE
            ### enthalpy calculation ###
            # gas phase
            # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
            # Cp mean list
            GaCpMeanList = calMeanHeatCapacityAtConstantPressure(
                comList, T_ReVa)
            # Cp mixture
            GaCpMeanMix = calMixtureHeatCapacityAtConstantPressure(
                MoFri, GaCpMeanList)
            # dimensionless analysis
            GaCpMeanMix_DiLeVa = rmtUtil.calDiLessValue(
                GaCpMeanMix, GaCpMeanMix0)
            # effective heat capacity - gas phase [kJ/kmol.K] | [J/mol.K]
            GaCpMeanMixEff = GaCpMeanMix*BeVoFr
            # dimensionless analysis
            GaCpMeanMixEff_DiLeVa = GaCpMeanMix_DiLeVa*BeVoFr

            # solid phase
            for r in range(rNo):
                # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
                # Cp mean list
                SoCpMeanList = calMeanHeatCapacityAtConstantPressure(
                    comList, Ts_r_ReVa[r])
                # Cp mixture
                SoCpMeanMix[r] = calMixtureHeatCapacityAtConstantPressure(
                    MoFrsi_r[r], SoCpMeanList)

                # effective heat capacity - solid phase [kJ/m^3.K]
                SoCpMeanMixEff_ReVa[r] = CosSp_r_ReVa[r] * \
                    SoCpMeanMix[r]*CaPo + (1-CaPo)*CaDe*CaSpHeCa

                # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
                # enthalpy change
                EnChList = np.array(
                    calEnthalpyChangeOfReaction(reactionListSorted, Ts_r_ReVa[r]))
                # heat of reaction at T [kJ/kmol] | [J/mol]
                HeReT = np.array(EnChList + StHeRe25)
                # overall heat of reaction [kJ/m^3.s]
                # exothermic reaction (negative sign)
                # endothermic sign (positive sign)
                OvHeReT[r] = np.dot(Ri_r[r, :], HeReT)

            # REVIEW
            if MODEL_SETTING['HeTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Prandtl Number
                # MW kg/mol -> g/mol
                # MiMoWe_Conv = 1000*MiMoWe
                PrNu = calPrNoEq1(
                    GaCpMeanMix, GaViMix, GaThCoMix, MiMoWe)
                # Nu number
                NuNu = calNuNoEq1(PrNu, ReNu)
                # heat transfer coefficient - gas/solid [J/m^2.s.K]
                HeTrCo = calHeatTransferCoefficientEq1(NuNu, GaThCoMix, PaDi)

            # REVIEW
            # heat transfer coefficient - medium side [J/m2.s.K]
            # hs = heat_transfer_coefficient_shell(T,Tv,Pv,Pa);
            # overall heat transfer coefficient [J/m2.s.K]
            # U = overall_heat_transfer_coefficient(hfs,kwall,do,di,L);
            # heat transfer coefficient - permeate side [J/m2.s.K]

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m^2/m^3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            # Ua = U*a
            # external heat [kJ/m^3.s]
            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T_ReVa, U, a, 'kJ/m^3.s')

            # NOTE
            # mass transfer between
            for i in range(compNo):
                ### gas phase ###
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # dimensionless analysis: real value
                Ci_f = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                # inward flux [kmol/m^2.s]
                MoFli_z[i] = MaTrCo[i]*Ci_f*(Ci_c - CosSpi_cat_DiLeVa[i])

            # total mass transfer between gas and solid phases [kmol/m^3]
            ToMaTrBeGaSo_z = np.sum(MoFli_z)*SpSuAr

            # NOTE
            # velocity from global concentration
            # check BC
            # if z == 0:
            #     # BC1
            #     constT_BC1 = (GaThCoEff)/(MoFl*GaCpMeanMix/1000)
            #     # next node
            #     T_f = T_z[z+1]
            #     # previous node
            #     T_b = (T0*dz + constT_BC1*T_f)/(dz + constT_BC1)
            # elif z == zNo - 1:
            #     # BC2
            #     # previous node
            #     T_b = T_z[z - 1]
            #     # next node
            #     T_f = 0
            # else:
            #     # interior nodes
            #     T_b = T_z[z-1]
            #     # next node
            #     T_f = T_z[z+1]

            # dxdt_v_T = (T_z[z] - T_b)/dz
            # # CoSp x 1000
            # # OvR x 1000
            # dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
            #                           ((1/T_z[z])*dxdt_P - (P_z[z]/T_z[z]**2)*dxdt_v_T) - ToMaTrBeGaSo_z*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            # FIXME
            # v_z[z+1] = dxdt_v*dz + v_z[z]
            # v_z[z+1] = v
            # FIXME
            v_z[z+1] = v_z[z]
            # dimensionless analysis
            v_z_DiLeVa = rmtUtil.calDiLessValue(v_z[z+1], vf)

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # bulk temperature [K]
            T_c = T_z[z]

            # universal index [j,i]
            # UISet = z*(rNo + 1)

            # NOTE
            # concentration [mol/m^3]
            for i in range(compNo):

                ### gas phase ###
                # mass balance (forward difference)
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]
                # check BC
                if z == 0 and solverMeshSet is True:
                    # NOTE
                    # BC1 (normal)
                    BC1_C_1 = PeNuMa0[i]*dz
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC1)
                elif z == 0 and solverMeshSet is False:
                    # NOTE
                    # BC1 (dense)
                    # i=0 is discretized based on inlet
                    # i=1
                    BC1_C_1 = PeNuMa0[i]*dzs[z]
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dzs[z], DIFF1_C_SET)
                    # d2Fdz2
                    # d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1)
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = 0
                    _zR_c = dzs[z]/dzs[z-1]
                    # dCdz = FiDiNonUniformDerivative1(
                    #     dFdz_C, dzs[z], DIFF1_C_SET, zR[z])
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1, _zR_c)
                    # FIXME
                    checkME = 0
                elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                    # NOTE
                    # dense section
                    # i=2,...,zNoNoDense-1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = dzs[z-2]/dzs[z-1]
                    _zR_c = dzs[z]/dzs[z-1]
                    #
                    dCdz = FiDiNonUniformDerivative1(
                        dFdz_C, dzs[z], DIFF1_C_SET, _zR_b)
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_G, _zR_c)
                    # FIXME
                    checkME = 0
                elif z == zNo - 1:
                    # NOTE
                    # BC2
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # forward difference
                    Ci_f = Ci_b
                    Ci_ff = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC2)
                else:
                    # NOTE
                    # normal sections
                    # interior nodes
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2] if z < zNo-2 else 0
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_G)

                # REVIEW
                # cal differentiate
                # backward difference
                # dCdz = (Ci_c - Ci_b)/(1*dz)
                # convective term
                _convectiveTerm = -1*v_z_DiLeVa*dCdz
                # central difference for dispersion
                # d2Cdz2 = (Ci_b - 2*Ci_c + Ci_f)/(dz**2)
                # dispersion term [kmol/m^3.s]
                _dispersionFluxC = (BeVoFr*GaDii_DiLeVa[i]/PeNuMa0[i])*d2Cdz2
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # inward flux [kmol/m^2.s]
                # MoFli_z[i] = MaTrCo[i]*(Ci_c - CosSpi_cat[i])
                _inwardFlux = (1/GaMaCoTe0[i])*MoFli_z[i]*SpSuAr
                # mass balance
                # convective, dispersion, inward flux
                # const
                _const1 = BeVoFr*(zf/vf)
                _const2 = 1/_const1
                #
                dxdt_F = _const2*(_convectiveTerm +
                                  _dispersionFluxC - _inwardFlux)
                dxdtMat[i][0][z] = dxdt_F

                ### solid phase ###
                # bulk concentration [kmol/m^3]
                # Ci_c
                # species concentration at different points of particle radius [rNo]
                # [Cs[3], Cs[2], Cs[1], Cs[0]]
                _Cs_r = CosSpi_r[:, i].flatten()
                # Cs[0], Cs[1], ...
                _Cs_r_Flip = np.flip(_Cs_r)
                # reaction term
                _ri_r = ri_r[:, i]
                # flip
                _ri_r_Flip = np.flip(_ri_r)

                # dimensionless analysis

                # loop
                _dCsdtiVarLoop = (
                    GaDii_DiLeVa[i], MaTrCo[i], _ri_r_Flip, Ci_c, CaPo, SoMaDiTe0[i], GaDii0[i], rf)

                # dC/dt list
                dCsdti = FiDiBuildCMatrix_DiLe(
                    compNo, PaRa, rNo, _Cs_r_Flip, _dCsdtiVarLoop, mode="default", fluxDir="rl")

                # const
                _const1 = CaPo*(rf**2/GaDii0[i])
                _const2 = 1/_const1
                #
                for r in range(rNo):
                    # update
                    dxdtMat[i][r+1][z] = _const2*dCsdti[r]

            # NOTE
            # energy balance
            # bulk temperature [K]
            # T_c
            # T_c = T_z[z]

            # temperature at different points of particle radius [rNo]
            # Ts[3], Ts[2], Ts[1], Ts[0]
            _Ts_r = Ts_r.flatten()
            # check BC
            if z == 0 and solverMeshSet is True:
                # BC1
                BC1_T_1 = PeNuHe0*dz
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC1)
            elif z == 0 and solverMeshSet is False:
                # BC1
                BC1_T_1 = PeNuHe0*dzs[z]
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dzs[z], DIFF1_T_SET)
                # d2Fdz2
                # d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF_T_SET_BC1)
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = 0
                _zR_c = dzs[z]/dzs[z-1]
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
                # FIXME
                checkME = 0
            elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                # NOTE
                # dense section
                # i=2,...,zNoNoDense-1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_bb, T_b, T_c, T_f, T_ff]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = dzs[z-2]/dzs[z-1]
                _zR_c = dzs[z]/dzs[z-1]
                #
                dTdz = FiDiNonUniformDerivative1(
                    dFdz_T, dzs[z], DIFF1_T_SET, _zR_b)
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
                # FIXME
                checkME = 0
            elif z == zNo - 1:
                # BC2
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # forward
                T_f = T_b
                T_ff = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC2)
            else:
                # interior nodes
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2] if z < zNo-2 else 0
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_G)

            # REVIEW
            # cal differentiate
            # backward difference
            # dTdz = (T_c - T_b)/(1*dz)
            # convective term
            _convectiveTerm = -1*v_z_DiLeVa*GaDe_DiLeVa*GaCpMeanMix_DiLeVa*dTdz
            # central difference
            # d2Tdz2 = (T_b - 2*T_c + T_f)/(dz**2)
            # dispersion flux [kJ/m^3.s]
            # _dispersionFluxT = (GaThCoEff*d2Tdz2)*1e-3
            _dispersionFluxT = ((1/PeNuHe0)*GaThCoEff_DiLeVa*d2Tdz2)*1
            # temperature in the catalyst surface [K]
            # Ts_cat
            # outward flux [kJ/m^2.s]
            _inwardFluxT = HeTrCo*SpSuAr*Tf*(_Ts_r[0] - T_c)*1e-3
            # total heat transfer between gas and solid [kJ/m^3.s]
            _heTrBeGaSoTerm = (1/GaHeCoTe0)*_inwardFluxT
            # heat exchange term [kJ/m^3.s] -> [no unit]
            _heatExchangeTerm = (1/GaHeCoTe0)*Qm
            # convective flux, diffusive flux, enthalpy of reaction, cooling heat
            # const
            _const1 = GaDe_DiLeVa*GaCpMeanMix_DiLeVa*BeVoFr*(zf/vf)
            _const2 = 1/_const1
            #
            dxdt_T = _const2*(_convectiveTerm + _dispersionFluxT +
                              _heTrBeGaSoTerm + _heatExchangeTerm)
            dxdtMat[indexT][0][z] = dxdt_T

            ### solid phase ###
            # _Ts_r
            # T[n], T[n-1], ..., T[0] => T[0],T[1], ...
            _Ts_r_Flip = np.flip(_Ts_r)

            # dC/dt list
            # convert
            # [J/s.m.K] => [kJ/s.m.K]
            SoThCoEff_Conv = CaPo*SoThCoMix0/1000
            # OvHeReT [kJ/m^3.s]
            OvHeReT_Conv = np.flip(-1*OvHeReT)
            # HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
            HeTrCo_Conv = HeTrCo/1000
            # var loop
            _dTsdtiVarLoop = (SoThCoEff_DiLeVa, HeTrCo_Conv,
                              OvHeReT_Conv, T_c, CaPo, SoHeDiTe0, SoThCoEff_Conv, rf)

            # dTs/dt list
            dTsdti = FiDiBuildTMatrix_DiLe(
                compNo, PaRa, rNo, _Ts_r_Flip, _dTsdtiVarLoop)

            # const
            _const1 = SoCpMeanMixEff_ReVa*Tf/SoHeDiTe0
            _const2 = 1/_const1
            #
            for r in range(rNo):
                # update
                dxdtMat[indexT][r+1][z] = _const2[r]*dTsdti[r]

        # NOTE
        # flat
        dxdt = dxdtMat.flatten().tolist()

        # print
        strTime = "time: {:.5f} seconds".format(t)
        # print(strTime)
        print(f"time: {t} seconds")

        return dxdt

# NOTE
# dynamic heterogenous modeling

    def runM8(self):
        """
        modeling case (dimensionless)
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static), Cci, Tc (static, for catalyst)
            CT, GaDe = f(P, T, n)
        numerical method: finite difference
        """
        # NOTE
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverIVPSet = solverConfig['ivp']
        solverMesh = solverConfig['mesh']
        solverMeshSet = True if solverMesh == "normal" else False

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        labelList = compList.copy()
        labelList.append("Temperature")
        # labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # diffusivity coefficient - gas phase [m^2/s]
        GaDii0 = self.modelInput['feed']['diffusivity']
        # gas viscosity [Pa.s]
        GaVii0 = self.modelInput['feed']['viscosity']
        # gas mixture viscosity [Pa.s]
        GaViMix0 = self.modelInput['feed']['mixture-viscosity']

        # thermal conductivity - gas phase [J/s.m.K]
        GaThCoi0 = self.modelInput['feed']['thermal-conductivity']
        # mixture thermal conductivity - gas phase [J/s.m.K]
        GaThCoMix0 = self.modelInput['feed']['mixture-thermal-conductivity']

        # REVIEW
        # domain length
        DoLe = 1
        # orthogonal collocation points in the r direction
        rNo = solverSetting['S2']['rNo']
        # mesh setting
        zMesh = solverSetting['T1']['zMesh']
        # number of nodes
        zNoNo = zMesh['zNoNo']
        # domain length section
        DoLeSe = zMesh['DoLeSe']
        # mesh refinement degree
        MeReDe = zMesh['MeReDe']
        # mesh installment
        if solverMeshSet is False:
            zMeshRes = FiDiMeshGenerator(zNoNo, DoLe, DoLeSe, MeReDe)
            # finite difference points
            dataXs = zMeshRes['data1']
            # dz lengths
            dzs = zMeshRes['data2']
            # finite difference point number
            zNo = zMeshRes['data3']
            # R ratio
            zR = zMeshRes['data4']
            # dz
            dz = zMeshRes['data5']
        else:
            # finite difference points in the z direction
            zNo = solverSetting['T1']['zNo']
            # length list [reactor length]
            dataXs = np.linspace(0, DoLe, zNo)
            # element size - dz [m]
            dz = DoLe/(zNo-1)
            # reset
            dzs = []
            zR = []

        ### calculation ###
        # mole fraction in the gas phase
        MoFri0 = np.array(rmtUtil.moleFractionFromConcentrationSpecies(SpCoi0))

        # mixture molecular weight [kg/mol]
        MiMoWe0 = rmtUtil.mixtureMolecularWeight(MoFri0, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe0 = calDensityIG(MiMoWe0, SpCo0*1000)

        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
        # Cp mean list
        GaCpMeanList0 = calMeanHeatCapacityAtConstantPressure(compList, T)
        # Cp mixture
        GaCpMeanMix0 = calMixtureHeatCapacityAtConstantPressure(
            MoFri0, GaCpMeanList0)

        # thermal diffusivity in the gas phase [m^2/s]
        GaThDi = calThermalDiffusivity(
            GaThCoMix0, GaDe0, GaCpMeanMix0, MiMoWe0)

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total number
        varNoConInSolid = varNoConInSolidBlock*zNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of layers
        noLayer = compNo + 1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = rNo + 1
        # number of var columns [i]
        varNoColumns = zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoRows, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayer - 1):
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # gas phase
                        IV2D[m][j][i] = SpCoi0[m]/np.max(SpCoi0)  # SpCoi0[m]
                    else:
                        # solid phase
                        # SpCoi0[m]/np.max(SpCoi0)  # SpCoi0[m]
                        # SpCoi0[m]/np.max(SpCoi0)  # SpCoi0[m]
                        IV2D[m][j][i] = 1e-6

        # temperature
        for i in range(varNoColumns):
            for j in range(varNoRows):
                # separate phase
                if j == 0:
                    # gas phase
                    IV2D[noLayer - 1][j][i] = 0  # T
                else:
                    # solid phase
                    IV2D[noLayer - 1][j][i] = 0  # T

        # flatten IV
        IV = IV2D.flatten()

        # print(f"IV: {IV}")

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted
        # reaction coefficient
        reactionStochCoeff = self.reactionStochCoeffList

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # REVIEW
        # solver setting
        # orthogonal collocation method
        OrCoClassSet = OrCoClass()
        OrCoClassSetRes = OrCoClassSet.buildMatrix()

        # NOTE
        ### dimensionless analysis ###
        # concentration [kmol/m^3]
        Cif = np.copy(SpCoi0)
        # total concentration
        Cf = SpCo0
        # temperature [K]
        Tf = T
        # superficial velocity [m/s]
        vf = SuGaVe0
        # length [m]
        zf = ReLe
        # diffusivity [m^2/s]
        Dif = np.copy(GaDii0)
        # heat capacity at constant pressure [J/mol.K] | [kJ/kmol.K]
        Cpif = np.copy(GaCpMeanList0)
        # mixture heat capacity [J/mol.K] | [kJ/kmol.K]
        Cpf = GaCpMeanMix0
        # radius
        rf = PaDi/2

        # gas phase
        # mass convective term - (list) [kmol/m^3.s]
        _Cif = Cif if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.repeat(
            np.max(Cif), compNo)
        GaMaCoTe0 = (vf/zf)*_Cif
        # mass diffusive term - (list)  [kmol/m^3.s]
        GaMaDiTe0 = (1/zf**2)*(_Cif*Dif)
        # heat convective term [kJ/m^3.s]
        GaHeCoTe0 = (GaDe0*vf*Tf*(Cpf/MiMoWe0)/zf)*1e-3
        # heat diffusive term [kJ/m^3.s]
        GaHeDiTe0 = (Tf*GaThCoMix0/zf**2)*1e-3

        # solid phase
        # mass diffusive term - (list)  [kmol/m^3.s]
        SoMaDiTe0 = (Dif*_Cif)/rf**2
        # heat diffusive term [kJ/m^3.s]
        SoHeDiTe0 = (GaThCoMix0*Tf/rf**2)*1e-3

        ### dimensionless numbers ###
        # Re Number
        ReNu0 = calReNoEq1(GaDe0, SuGaVe0, PaDi, GaViMix0)
        # Sc Number
        ScNu0 = calScNoEq1(GaDe0, GaViMix0, GaDii0)
        # Sh Number (choose method)
        ShNu0 = calShNoEq1(ScNu0, ReNu0, CONST_EQ_Sh['Frossling'])
        # Prandtl Number
        PrNu0 = calPrNoEq1(GaCpMeanMix0, GaViMix0, GaThCoMix0, MiMoWe0)
        # Nu number
        NuNu0 = calNuNoEq1(PrNu0, ReNu0)
        # Strouhal number
        StNu = 1
        # Peclet number - mass transfer
        PeNuMa0 = (vf*zf)/Dif
        # Peclet number - heat transfer
        PeNuHe0 = (zf*GaDe0*(Cpf/MiMoWe0)*vf)/GaThCoMix0

        ### transfer coefficient ###
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = calMassTransferCoefficientEq1(ShNu0, GaDii0, PaDi)
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = calHeatTransferCoefficientEq1(NuNu0, GaThCoMix0, PaDi)

        # fun parameters
        FunParam = {
            "compList": compList,
            "const": {
                "CrSeAr": CrSeAr,
                "MoWei": MoWei,
                "StHeRe25": StHeRe25,
                "GaMiVi": GaViMix0,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T,
                "SuGaVe0": SuGaVe0,
                "GaDii0": GaDii0,
                "GaThCoi0": GaThCoi0,
                "GaVii0": GaVii0,
                "GaDe0": GaDe0,
                "GaCpMeanMix0": GaCpMeanMix0,
                "GaThCoMix0": GaThCoMix0
            },
            "meshSetting": {
                "solverMesh": solverMesh,
                "solverMeshSet": solverMeshSet,
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz,
                "dzs": dzs,
                "zR": zR,
                "zNoNo": zNoNo
            },
            "solverSetting": {
                "dFdz": solverSetting['T1']['dFdz'],
                "d2Fdz2": solverSetting['T1']['d2Fdz2'],
                "dTdz": solverSetting['T1']['dTdz'],
                "d2Tdz2": solverSetting['T1']['d2Tdz2'],
                "OrCoClassSetRes": OrCoClassSetRes
            },
            "reactionRateExpr": reactionRateExpr

        }

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "zf": zf,
            "Dif": Dif,
            "Cpif": Cpif,
            "Cpf": Cpf,
            "rf": rf,
            "GaMaCoTe0": GaMaCoTe0,
            "GaMaDiTe0": GaMaDiTe0,
            "GaHeCoTe0": GaHeCoTe0,
            "GaHeDiTe0": GaHeDiTe0,
            "ReNu0": ReNu0,
            "ScNu0": ScNu0,
            "ShNu0": ShNu0,
            "PrNu0": PrNu0,
            "PeNuMa0": PeNuMa0,
            "PeNuHe0": PeNuHe0,
            "MaTrCo": MaTrCo,
            "HeTrCo": HeTrCo,
            "SoMaDiTe0": SoMaDiTe0,
            "SoHeDiTe0": SoHeDiTe0
        }

        # time span
        tNo = solverSetting['S2']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['S2']['timesNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((varNo, tNo, zNo))
        #

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            # method [1]: LSODA, [2]: BDF, [3]: Radau
            # options
            solverOptions = {
                "atol": 1e-7
            }

            sol = solve_ivp(PackedBedReactorClass.modelEquationM8,
                            t, IV, method=solverIVP, t_eval=times,  args=(reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams))

            # ode result
            successStatus = sol.success
            # check
            if successStatus is False:
                raise

            # time interval
            dataTime = sol.t
            # all results
            # components, temperature layers
            dataYs = sol.y

            # std format
            dataYs_Reshaped = np.reshape(
                dataYs[:, -1], (noLayer, varNoRows, varNoColumns))

            # component concentration [kmol/m^3]
            # Ci and Cs
            # dataYs1 = dataYs[0:varNoCon, -1]
            # 3d matrix
            # dataYs1_Reshaped = np.reshape(
            #     dataYs1, (compNo, varNoRows, varNoColumns))

            dataYs1_Reshaped = dataYs_Reshaped[:-1]

            # gas phase
            dataYs1GasPhase = dataYs1_Reshaped[:, 0, :]
            # solid phase
            dataYs1SolidPhase = dataYs1_Reshaped[:, 1:, :]

            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1GasPhase, axis=0)
            dataYs1_MoFri = dataYs1GasPhase/dataYs1_Ctot

            # temperature - 2d matrix
            # dataYs2 = np.array([dataYs[varNoCon:varNoLayerT, -1]])
            # 2d matrix
            # dataYs2_Reshaped = np.reshape(
            #     dataYs2, (1, varNoRows, varNoColumns))

            dataYs2_Reshaped = dataYs_Reshaped[indexTemp]
            # gas phase
            dataYs2GasPhase = dataYs2_Reshaped[0, :].reshape((1, zNo))
            # solid phase
            dataYs2SolidPhase = dataYs2_Reshaped[1:, :]

            # combine
            _dataYs = np.concatenate(
                (dataYs1_MoFri, dataYs2GasPhase), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCon": dataYs1GasPhase,
                "dataYTemp": dataYs2GasPhase,
                "dataYs": _dataYs,
                "dataYCons": dataYs1SolidPhase,
                "dataYTemps": dataYs2SolidPhase,
            })

            for m in range(varNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # NOTE
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        ssdataLists = [ssdataList[0:compNo],
                       ssdataList[indexTemp]]
        # subplot result
        # pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

        # plot info
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # display result at specific time
        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataYs']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(dataLists, "Reactor Length (m)",
                                "Concentration (mol/m^3)", plotTitle, ssdataLists)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _name = labelList[i] + " at t=" + str(opTSpan[t+1])

                _labelNameTime.append(_name)

            dataList = pltc.plots2DSetDataList(XYList, _labelNameTime)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select items
        # indices = [0, 2, -1]
        # selected_elements = [_dataListsLoop[index] for index in indices]
        # select datalist
        _dataListsSelected = selectFromListByIndex([1, -1], _dataListsLoop)

        # subplot result
        # pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
        #                 "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM8(t, y, reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams):
        """
            model [dynamic modeling]
            mass, energy, and momentum balance equations
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
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]
                    meshSetting:
                        solverMesh: mesh installment
                        solverMeshSet: 
                            true: normal
                            false: mesh refinement
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                        dzs: differential length list [-]
                        zR: z ratio
                        zNoNo: number of nodes in the dense and normal sections
                    solverSetting:
                        OrCoClassSetRes: constants of OC methods
                    reactionRateExpr: reaction rate expressions
                DimensionlessAnalysisParams:
                    Cif: feed concentration [kmol/m^3]
                    Tf: feed temperature
                    vf: feed superficial velocity [m/s]
                    zf: domain length [m]
                    Dif: diffusivity coefficient of component [m^2/s]
                    Cpif: feed heat capacity at constat pressure [kJ/kmol.K] | [J/mol.K]
                    rf: particle radius [m]
                    GaMaCoTe0: feed mass convective term of gas phase [kmol/m^3.s]
                    GaMaDiTe0: feed mass diffusive term of gas phase [kmol/m^3.s]
                    GaHeCoTe0: feed heat convective term of gas phase [kJ/m^3.s]
                    GaHeDiTe0, feed heat diffusive term of gas phase [kJ/m^3.s]
                    SoMaDiTe0: feed mass diffusive term of solid phase [kmol/m^3.s]
                    SoHeDiTe0: feed heat diffusive term of solid phase [kJ/m^3.s]
                    ReNu0: Reynolds number
                    ScNu0: Schmidt number
                    ShNu0: Sherwood number
                    PrNu0: Prandtl number
                    PeNuMa0: mass Peclet number
                    PeNuHe0: heat Peclet number 
                    MaTrCo: mass transfer coefficient - gas/solid [m/s]
                    HeTrCo: heat transfer coefficient - gas/solid [J/m^2.s.K]
        """
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

        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']
        # catalyst porosity
        CaPo = ReSpec['CaPo']
        # catalyst tortuosity
        CaTo = ReSpec['CaTo']
        # catalyst thermal conductivity [J/K.m.s]
        CaThCo = ReSpec['CaThCo']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']
        # inlet superficial velocity [m/s]
        # SuGaVe0 = constBC1['SuGaVe0']
        # inlet diffusivity coefficient [m^2]
        GaDii0 = constBC1['GaDii0']
        # inlet gas thermal conductivity [J/s.m.K]
        GaThCoi0 = constBC1['GaThCoi0']
        # gas viscosity
        GaVii0 = constBC1['GaVii0']
        # gas density [kg/m^3]
        GaDe0 = constBC1['GaDe0']
        # heat capacity at constant pressure [kJ/kmol.K] | [J/mol.K]
        GaCpMeanMix0 = constBC1['GaCpMeanMix0']
        # gas thermal conductivity [J/s.m.K]
        GaThCoMix0 = constBC1['GaThCoMix0']

        # mesh setting
        meshSetting = FunParam['meshSetting']
        # mesh installment
        solverMesh = meshSetting['solverMesh']
        # mesh refinement
        solverMeshSet = meshSetting['solverMeshSet']
        # number of layers
        noLayer = meshSetting['noLayer']
        # var no in each layer
        varNoLayer = meshSetting['varNoLayer']
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = meshSetting['varNoLayerT']
        # number of var rows [j]
        varNoRows = meshSetting['varNoRows']
        # number of var columns [i]
        varNoColumns = meshSetting['varNoColumns']
        # rNo
        rNo = meshSetting['rNo']
        # zNo
        zNo = meshSetting['zNo']
        # dz [m]
        dz = meshSetting['dz']
        # dzs [m]/[-]
        dzs = meshSetting['dzs']
        # R ratio
        zR = meshSetting['zR']
        # number of nodes in the dense and normal sections
        zNoNo = meshSetting['zNoNo']
        # dense
        zNoNoDense = zNoNo[0]
        # normal
        zNoNoNormal = zNoNo[1]

        # solver setting
        solverSetting = FunParam['solverSetting']
        # mass balance equation
        DIFF1_C_SET = solverSetting['dFdz']
        DIFF2_C_SET_BC1 = solverSetting['d2Fdz2']['BC1']
        DIFF2_C_SET_BC2 = solverSetting['d2Fdz2']['BC2']
        DIFF2_C_SET_G = solverSetting['d2Fdz2']['G']
        # energy balance equation
        DIFF1_T_SET = solverSetting['dTdz']
        DIFF2_T_SET_BC1 = solverSetting['d2Tdz2']['BC1']
        DIFF2_T_SET_BC2 = solverSetting['d2Tdz2']['BC2']
        DIFF2_T_SET_G = solverSetting['d2Tdz2']['G']
        # number of collocation points
        ocN = solverSetting['OrCoClassSetRes']['N']
        ocXc = solverSetting['OrCoClassSetRes']['Xc']
        ocA = solverSetting['OrCoClassSetRes']['A']
        ocB = solverSetting['OrCoClassSetRes']['B']
        ocQ = solverSetting['OrCoClassSetRes']['Q']

        # init OrCoCatParticle
        OrCoCatParticleClassSet = OrCoCatParticleClass(
            ocXc, ocN, ocQ, ocA, ocB, varNo)

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # dimensionless analysis params

        #  feed concentration [kmol/m^3]
        Cif = DimensionlessAnalysisParams['Cif']
        # feed temperature
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # domain length [m]
        zf = DimensionlessAnalysisParams['zf']
        # particle radius [m]
        rf = DimensionlessAnalysisParams['rf']
        # diffusivity coefficient of component [m^2/s]
        Dif = DimensionlessAnalysisParams['Dif']
        # feed heat capacity at constat pressure
        Cpif = DimensionlessAnalysisParams['Cpif']
        # feed mass convective term of gas phase [kmol/m^3.s]
        GaMaCoTe0 = DimensionlessAnalysisParams['GaMaCoTe0']
        # feed mass diffusive term of gas phase [kmol/m^3.s]
        GaMaDiTe0 = DimensionlessAnalysisParams['GaMaDiTe0']
        # feed heat convective term of gas phase [kJ/m^3.s]
        GaHeCoTe0 = DimensionlessAnalysisParams['GaHeCoTe0']
        # feed heat diffusive term of gas phase [kJ/m^3.s]
        GaHeDiTe0 = DimensionlessAnalysisParams['GaHeDiTe0']
        # feed mass diffusive term of solid phase [kmol/m^3.s]
        SoMaDiTe0 = DimensionlessAnalysisParams['SoMaDiTe0']
        # feed heat diffusive term of solid phase [kJ/m^3.s]
        SoHeDiTe0 = DimensionlessAnalysisParams['SoHeDiTe0']
        # Reynolds number
        ReNu = DimensionlessAnalysisParams['ReNu0']
        # Schmidt number
        ScNu = DimensionlessAnalysisParams['ScNu0']
        # Sherwood number
        ShNu = DimensionlessAnalysisParams['ShNu0']
        # Prandtl number
        PrNu = DimensionlessAnalysisParams['PrNu0']
        # mass Peclet number
        PeNuMa0 = DimensionlessAnalysisParams['PeNuMa0']
        # heat Peclet number
        PeNuHe0 = DimensionlessAnalysisParams['PeNuHe0']
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = DimensionlessAnalysisParams['MaTrCo']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = DimensionlessAnalysisParams['HeTrCo']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexV = indexP + 1

        # calculate
        # particle radius
        PaRa = PaDi/2
        # specific surface area exposed to the free fluid [m^2/m^3]
        SpSuAr = (3/PaRa)*(1 - BeVoFr)

        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # interstitial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate in the solid phase
        Ri_z = np.zeros((zNo, reactionListNo))
        Ri_zr = np.zeros((zNo, rNo, reactionListNo))
        Ri_r = np.zeros((rNo, reactionListNo))
        # reaction rate
        # ri = np.zeros(compNo) # deprecate
        # ri0 = np.zeros(compNo) # deprecate
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        # overall reaction
        OvR = np.zeros(rNo)
        # overall enthalpy
        OvHeReT = np.zeros(rNo)
        # heat capacity at constant pressure
        SoCpMeanMix = np.zeros(rNo)
        # effective heat capacity at constant pressure
        SoCpMeanMixEff = np.zeros(rNo)
        # dimensionless analysis
        SoCpMeanMixEff_ReVa = np.zeros(rNo)

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # superficial gas velocity [m/s]
        v_z = np.zeros(zNo + 1)
        v_z[0] = SuGaVe0

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (noLayer, varNoRows, varNoColumns))

        # all species concentration in gas & solid phase
        SpCo_mz = np.zeros((noLayer - 1, varNoRows, varNoColumns))
        # all species concentration in gas phase [kmol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        # all species concentration in solid phase (catalyst) [kmol/m^3]
        SpCosi_mzr = np.zeros((compNo, rNo, zNo))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCo_mz[m] = _SpCoi
        # concentration in the gas phase [kmol/m^3]
        for m in range(compNo):
            for j in range(varNoRows):
                if j == 0:
                    # gas phase
                    SpCoi_z[m, :] = SpCo_mz[m, j, :]
                else:
                    # solid phase
                    SpCosi_mzr[m, j-1, :] = SpCo_mz[m, j, :]

        # species concentration in gas phase [kmol/m^3]
        CoSpi = np.zeros(compNo)
        # dimensionless analysis
        CoSpi_ReVa = np.zeros(compNo)
        # total concentration [kmol/m^3]
        CoSp = 0
        # species concentration in solid phase (catalyst) [kmol/m^3]
        # shape
        CosSpiMatShape = (rNo, compNo)
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)
        # total concentration in the solid phase [kmol/m^3]
        CosSp_r = np.zeros(rNo)

        # flux
        MoFli_z = np.zeros(compNo)

        # NOTE
        # temperature [K]
        T_mz = np.zeros((varNoRows, varNoColumns))
        T_mz = yLoop[noLayer - 1]
        # temperature in the gas phase
        T_z = np.zeros(zNo)
        T_z = T_mz[0, :]
        # temperature in solid phase
        Ts_z = np.zeros((rNo, zNo))
        Ts_z = T_mz[1:]
        # temperature in the solid phase
        Ts_r = np.zeros(rNo)

        # diff/dt
        # dxdt = []
        # matrix
        # dxdtMat = np.zeros((varNo, zNo))
        dxdtMat = np.zeros((noLayer, varNoRows, varNoColumns))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(varNoColumns):
            ## block ##

            # concentration species in the gas phase [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = max(_SpCoi_z, CONST.EPS_CONST)
                # REVIEW
                # dimensionless analysis: real value
                SpCoi0_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                CoSpi_ReVa[i] = rmtUtil.calRealDiLessValue(
                    CoSpi[i], SpCoi0_Set)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)
            # dimensionless analysis: real value
            CoSp_ReVa = np.sum(CoSpi_ReVa)

            # FIXME
            # concentration species in the solid phase [kmol/m^3]
            # display concentration list in each oc point (rNo)
            for i in range(compNo):
                for r in range(rNo):
                    _CosSpi_z = SpCosi_mzr[i][r][z]
                    CosSpi_r[r][i] = max(_CosSpi_z, CONST.EPS_CONST)
                    # REVIEW
                    # dimensionless analysis: real value
                    SpCoi0_r_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                        SpCoi0)
                    CosSpi_r_ReVa[r][i] = rmtUtil.calRealDiLessValue(
                        CosSpi_r[r][i], SpCoi0_r_Set)

            # total concentration in the solid phase [kmol/m^3]
            CosSp_r = np.sum(CosSpi_r, axis=1).reshape((rNo, 1))
            # dimensionless analysis: real value
            CosSp_r_ReVa = np.sum(CosSpi_r_ReVa, axis=1).reshape((rNo, 1))

            # concentration in the outer surface of the catalyst [kmol/m^3]
            CosSpi_cat = CosSpi_r[0]
            # dimensionless analysis
            CosSpi_cat_DiLeVa = CosSpi_r[0, :]

            # temperature [K]
            T = T_z[z]
            T_ReVa = rmtUtil.calRealDiLessValue(T, T0, "TEMP")
            # temperature in the solid phase (for each point)
            # Ts[3], Ts[2], Ts[1], Ts[0]
            Ts_r = Ts_z[:, z]
            Ts_r_ReVa = rmtUtil.calRealDiLessValue(Ts_r, T0, "TEMP")

            # pressure [Pa]
            P = P_z[z]

            # FIXME
            # velocity
            # dimensionless value
            # v = v_z[z]
            v = 1

            ## calculate ##
            # mole fraction in the gas phase
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi_ReVa))

            # mole fraction in the solid phase
            # MoFrsi_r0 = CosSpi_r/CosSp_r
            MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
                CosSpi_r_ReVa)

            # TODO
            # dv/dz
            # gas velocity based on interstitial velocity [m/s]
            # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            # SuGaVe = InGaVe*BeVoFr
            # from ode eq. dv/dz
            SuGaVe = v
            # dimensionless analysis
            SuGaVe_ReVa = rmtUtil.calRealDiLessValue(SuGaVe, SuGaVe0)

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = calMolarFlowRate(CoSp_ReVa, SuGaVe_ReVa, CrSeAr)
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp_ReVa*1000)
            # GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)
            # dimensionless value
            GaDe_DiLeVa = rmtUtil.calDiLessValue(GaDe, GaDe0)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe_ReVa/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe_ReVa**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # REVIEW
            # FIXME
            # viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaVii = GaVii0 if MODEL_SETTING['GaVii'] == "FIX" else calTest()
            # mixture viscosity in the gas phase [Pa.s] | [kg/m.s]
            # FIXME
            GaViMix = 2.5e-5  # f(yi,GaVi,MWs);
            # kinematic viscosity in the gas phase [m^2/s]
            GaKiViMix = GaViMix/GaDe

            # REVIEW
            # FIXME
            # solid gas thermal conductivity
            SoThCoMix0 = GaThCoMix0
            # add loop for each r point/constant
            # catalyst thermal conductivity [J/s.m.K]
            # CaThCo
            # membrane wall thermal conductivity [J/s.m.K]
            MeThCo = 1
            # thermal conductivity - gas phase [J/s.m.K]
            # GaThCoi = np.zeros(compNo)  # f(T);
            GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest(
            )
            # dimensionless
            GaThCoi_DiLe = GaThCoi/GaThCoi0
            # FIXME
            # mixture thermal conductivity - gas phase [J/s.m.K]
            GaThCoMix = GaThCoMix0
            # dimensionless analysis
            GaThCoMix_DiLeVa = GaThCoMix/GaThCoMix0
            # thermal conductivity - solid phase [J/s.m.K]
            # assume the same as gas phase
            # SoThCoi = np.zeros(compNo)  # f(T);
            SoThCoi = GaThCoi
            # mixture thermal conductivity - solid phase [J/s.m.K]
            SoThCoMix = GaThCoMix0
            # dimensionless analysis
            SoThCoMix_DiLeVa = SoThCoMix/SoThCoMix0
            # effective thermal conductivity - gas phase [J/s.m.K]
            # GaThCoEff = BeVoFr*GaThCoMix + (1 - BeVoFr)*CaThCo
            GaThCoEff = BeVoFr*GaThCoMix
            # dimensionless analysis
            GaThCoEff_DiLeVa = BeVoFr*GaThCoMix_DiLeVa
            # FIXME
            # effective thermal conductivity - solid phase [J/s.m.K]
            # assume identical to gas phase
            # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
            # SoThCoEff = SoThCoMix*((1 - CaPo)/CaTo)
            SoThCoEff = CaPo*SoThCoMix
            # dimensionless analysis
            # SoThCoEff_DiLeVa = GaThCoMix_DiLeVa*((1 - CaPo)/CaTo)
            SoThCoEff_DiLeVa = CaPo*SoThCoMix_DiLeVa

            # REVIEW
            # diffusivity coefficient - gas phase [m^2/s]
            GaDii = GaDii0 if MODEL_SETTING['GaDii'] == "FIX" else calTest()
            # dimensionless analysis
            GaDii_DiLeVa = GaDii/GaDii0
            # effective diffusivity coefficient - gas phase
            GaDiiEff = GaDii*BeVoFr
            # dimensionless analysis
            GaDiiEff_DiLeVa = GaDiiEff/GaDii0
            # effective diffusivity - solid phase [m^2/s]
            SoDiiEff = (CaPo/CaTo)*GaDii
            # dimensionless analysis
            SoDiiEff_DiLe = (CaPo/CaTo)*GaDii_DiLeVa

            # REVIEW
            if MODEL_SETTING['MaTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Re Number
                ReNu = calReNoEq1(GaDe, SuGaVe, PaDi, GaViMix)
                # Sc Number
                ScNu = calScNoEq1(GaDe, GaViMix, GaDii)
                # Sh Number (choose method)
                ShNu = calShNoEq1(ScNu, ReNu, CONST_EQ_Sh['Frossling'])

                # mass transfer coefficient - gas/solid [m/s]
                MaTrCo = calMassTransferCoefficientEq1(ShNu, GaDii, PaDi)

            # NOTE
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
            for r in range(rNo):
                # loop
                loopVars0 = (Ts_r_ReVa[r], P_z[z],
                             MoFrsi_r[r], CosSpi_r_ReVa[r])

                # component formation rate [mol/m^3.s]
                # check unit
                r0 = np.array(reactionRateExe(
                    loopVars0, varisSet, ratesSet))

                # loop
                Ri_zr[z, r, :] = r0
                Ri_r[r, :] = r0

                # component formation rate [kmol/m^3.s]
                ri_r[r] = componentFormationRate(
                    compNo, comList, reactionStochCoeff, Ri_r[r])

                # overall formation rate [kmol/m^3.s]
                OvR[r] = np.sum(ri_r[r])

            # NOTE
            ### enthalpy calculation ###
            # gas phase
            # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
            # Cp mean list
            GaCpMeanList = calMeanHeatCapacityAtConstantPressure(
                comList, T_ReVa)
            # Cp mixture
            GaCpMeanMix = calMixtureHeatCapacityAtConstantPressure(
                MoFri, GaCpMeanList)
            # dimensionless analysis
            GaCpMeanMix_DiLeVa = rmtUtil.calDiLessValue(
                GaCpMeanMix, GaCpMeanMix0)
            # effective heat capacity - gas phase [kJ/kmol.K] | [J/mol.K]
            GaCpMeanMixEff = GaCpMeanMix*BeVoFr
            # dimensionless analysis
            GaCpMeanMixEff_DiLeVa = GaCpMeanMix_DiLeVa*BeVoFr

            # solid phase
            for r in range(rNo):
                # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
                # Cp mean list
                SoCpMeanList = calMeanHeatCapacityAtConstantPressure(
                    comList, Ts_r[r])
                # Cp mixture
                SoCpMeanMix[r] = calMixtureHeatCapacityAtConstantPressure(
                    MoFrsi_r[r], SoCpMeanList)

                # effective heat capacity - solid phase [kJ/m^3.K]
                SoCpMeanMixEff_ReVa[r] = CosSp_r_ReVa[r] * \
                    SoCpMeanMix[r]*CaPo + (1-CaPo)*CaDe*CaSpHeCa

                # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
                # enthalpy change
                EnChList = np.array(
                    calEnthalpyChangeOfReaction(reactionListSorted, Ts_r[r]))
                # heat of reaction at T [kJ/kmol] | [J/mol]
                HeReT = np.array(EnChList + StHeRe25)
                # overall heat of reaction [kJ/m^3.s]
                # exothermic reaction (negative sign)
                # endothermic sign (positive sign)
                OvHeReT[r] = np.dot(Ri_r[r, :], HeReT)

            # REVIEW
            if MODEL_SETTING['HeTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Prandtl Number
                # MW kg/mol -> g/mol
                # MiMoWe_Conv = 1000*MiMoWe
                PrNu = calPrNoEq1(
                    GaCpMeanMix, GaViMix, GaThCoMix, MiMoWe)
                # Nu number
                NuNu = calNuNoEq1(PrNu, ReNu)
                # heat transfer coefficient - gas/solid [J/m^2.s.K]
                HeTrCo = calHeatTransferCoefficientEq1(NuNu, GaThCoMix, PaDi)

            # REVIEW
            # heat transfer coefficient - medium side [J/m2.s.K]
            # hs = heat_transfer_coefficient_shell(T,Tv,Pv,Pa);
            # overall heat transfer coefficient [J/m2.s.K]
            # U = overall_heat_transfer_coefficient(hfs,kwall,do,di,L);
            # heat transfer coefficient - permeate side [J/m2.s.K]

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m^2/m^3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            # Ua = U*a
            # external heat [kJ/m^3.s]
            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T_ReVa, U, a, 'kJ/m^3.s')

            # NOTE
            # # mass transfer between
            # for i in range(compNo):
            #     ### gas phase ###
            #     # mass balance (forward difference)
            #     # concentration [kmol/m^3]
            #     # central
            #     Ci_c = SpCoi_z[i][z]
            #     # concentration in the catalyst surface [kmol/m^3]
            #     # CosSpi_cat
            #     # dimensionless analysis: real value
            #     Ci_f = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
            #         SpCoi0)
            #     # inward flux [kmol/m^2.s]
            #     MoFli_z[i] = MaTrCo[i]*Ci_f*(Ci_c - CosSpi_cat_DiLeVa[i])

            # # total mass transfer between gas and solid phases [kmol/m^3]
            # ToMaTrBeGaSo_z = np.sum(MoFli_z)*SpSuAr

            # NOTE
            # velocity from global concentration
            # check BC
            # if z == 0:
            #     # BC1
            #     constT_BC1 = (GaThCoEff)/(MoFl*GaCpMeanMix/1000)
            #     # next node
            #     T_f = T_z[z+1]
            #     # previous node
            #     T_b = (T0*dz + constT_BC1*T_f)/(dz + constT_BC1)
            # elif z == zNo - 1:
            #     # BC2
            #     # previous node
            #     T_b = T_z[z - 1]
            #     # next node
            #     T_f = 0
            # else:
            #     # interior nodes
            #     T_b = T_z[z-1]
            #     # next node
            #     T_f = T_z[z+1]

            # dxdt_v_T = (T_z[z] - T_b)/dz
            # # CoSp x 1000
            # # OvR x 1000
            # dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
            #                           ((1/T_z[z])*dxdt_P - (P_z[z]/T_z[z]**2)*dxdt_v_T) - ToMaTrBeGaSo_z*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            # FIXME
            # v_z[z+1] = dxdt_v*dz + v_z[z]
            # v_z[z+1] = v
            # FIXME
            v_z[z+1] = v_z[z]
            # dimensionless analysis
            v_z_DiLeVa = rmtUtil.calDiLessValue(v_z[z+1], vf)

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # bulk temperature [K]
            T_c = T_z[z]

            # REVIEW
            # gas-solid interface BC
            # concentration [m/s]*[m^2/s]=[1/m]
            # betaC = PaRa*(MaTrCo/SoDiiEff)
            # temperature
            # betaT = -1*((HeTrCo*PaRa)/SoThCoEff)

            # universal index [j,i]
            # UISet = z*(rNo + 1)

            # NOTE
            # concentration [mol/m^3]
            for i in range(compNo):
                # concentration [kmol/m^3]
                # central
                Ci_c = SpCoi_z[i][z]

                # REVIEW
                ### solid phase ###
                # bulk concentration [kmol/m^3]
                # Ci_c
                # species concentration at different points of particle radius [rNo]
                # [Cs[3], Cs[2], Cs[1], Cs[0]]
                _Cs_r = CosSpi_r[:, i].flatten()
                # Cs[0], Cs[1], ...
                _Cs_r_Flip = np.flip(_Cs_r)

                # dimensionless analysis
                # beta
                # const
                _alpha = rf/GaDii0[i]
                _beta = MaTrCo[i]/GaDii_DiLeVa[i]
                _Cs_r_interface = _alpha*_beta
                _Ri = (1/SoMaDiTe0[i])*(1 - CaPo)*ri_r[:, i]
                # updated concentration gas-solid interface
                # shape(rNo,1)
                _Cs_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                    _Cs_r, Ci_c, _Cs_r_interface)

                # # dC/dt list
                dCsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                    _Cs_r_Updated, SoDiiEff_DiLe[i], _Ri)

                # const
                _const1 = CaPo*(rf**2/GaDii0[i])
                _const2 = 1/_const1
                #
                for r in range(rNo):
                    # update
                    dxdtMat[i][r+1][z] = _const2*dCsdti[r]

                # concentration [kmol/m^3]
                # central
                # Ci_c = SpCoi_z[i][z]
                # concentration in the catalyst surface [kmol/m^3]
                CosSpi_cat_gas = _Cs_r_Updated[-1]
                # dimensionless analysis: real value
                Ci_f = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                # inward flux [kmol/m^2.s]
                MoFli_z[i] = MaTrCo[i]*Ci_f*(Ci_c - CosSpi_cat_gas)

                # REVIEW
                ### gas phase ###
                # check BC
                if z == 0 and solverMeshSet is True:
                    # NOTE
                    # BC1 (normal)
                    BC1_C_1 = PeNuMa0[i]*dz
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC1)
                elif z == 0 and solverMeshSet is False:
                    # NOTE
                    # BC1 (dense)
                    # i=0 is discretized based on inlet
                    # i=1
                    BC1_C_1 = PeNuMa0[i]*dzs[z]
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dzs[z], DIFF1_C_SET)
                    # d2Fdz2
                    # d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1)
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = 0
                    _zR_c = dzs[z]/dzs[z-1]
                    # dCdz = FiDiNonUniformDerivative1(
                    #     dFdz_C, dzs[z], DIFF1_C_SET, zR[z])
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1, _zR_c)
                elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                    # NOTE
                    # dense section
                    # i=2,...,zNoNoDense-1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = dzs[z-2]/dzs[z-1]
                    _zR_c = dzs[z]/dzs[z-1]
                    #
                    dCdz = FiDiNonUniformDerivative1(
                        dFdz_C, dzs[z], DIFF1_C_SET, _zR_b)
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_G, _zR_c)
                elif z == zNo - 1:
                    # NOTE
                    # BC2
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # forward difference
                    Ci_f = Ci_b
                    Ci_ff = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC2)
                else:
                    # NOTE
                    # normal sections
                    # interior nodes
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2] if z < zNo-2 else 0
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_G)

                # REVIEW
                # cal differentiate
                # backward difference
                # dCdz = (Ci_c - Ci_b)/(1*dz)
                # convective term
                _convectiveTerm = -1*v_z_DiLeVa*dCdz
                # central difference for dispersion
                # d2Cdz2 = (Ci_b - 2*Ci_c + Ci_f)/(dz**2)
                # dispersion term [kmol/m^3.s]
                _dispersionFluxC = (BeVoFr*GaDii_DiLeVa[i]/PeNuMa0[i])*d2Cdz2
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # inward flux [kmol/m^2.s]
                # MoFli_z[i] = MaTrCo[i]*(Ci_c - CosSpi_cat[i])
                _inwardFlux = (1/GaMaCoTe0[i])*MoFli_z[i]*SpSuAr
                # mass balance
                # convective, dispersion, inward flux
                # const
                _const1 = BeVoFr*(zf/vf)
                _const2 = 1/_const1
                #
                dxdt_F = _const2*(_convectiveTerm +
                                  _dispersionFluxC - _inwardFlux)
                dxdtMat[i][0][z] = dxdt_F

            # NOTE
            # energy balance
            # bulk temperature [K]
            # T_c
            # T_c = T_z[z]
            # REVIEW
            ### solid phase ###
            # temperature at different points of particle radius [rNo]
            # Ts[3], Ts[2], Ts[1], Ts[0]
            _Ts_r = Ts_r.flatten()
            # _Ts_r
            # T[n], T[n-1], ..., T[0] => T[0],T[1], ...
            _Ts_r_Flip = np.flip(_Ts_r)

            # dC/dt list
            # convert
            # [J/s.m.K] => [kJ/s.m.K]
            SoThCoEff_Conv = CaPo*SoThCoMix0/1000
            # OvHeReT [kJ/m^3.s]
            OvHeReT_Conv = -1*OvHeReT
            # HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
            HeTrCo_Conv = HeTrCo/1000

            # loop vars
            _alpha = rf/SoThCoEff_Conv
            _beta = -1*HeTrCo_Conv/SoThCoEff_DiLeVa
            _Ts_r_interfaceVar = _alpha*_beta
            _H = (1/SoHeDiTe0)*(1 - CaPo)*OvHeReT_Conv
            # T[n], T[n-1], ..., T[0]
            # updated temperature gas--solid interface
            _Ts_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                _Ts_r, T_c, _Ts_r_interfaceVar)

            # dTs/dt list
            dTsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                _Ts_r_Updated, SoThCoEff_DiLeVa, _H)

            # const
            _const1 = SoCpMeanMixEff_ReVa*Tf/SoHeDiTe0
            _const2 = 1/_const1
            #
            for r in range(rNo):
                # update
                dxdtMat[indexT][r+1][z] = _const2[r]*dTsdti[r]

            # updated temperature in the gas-solid interface
            Ts_r_cat_gas = _Ts_r_Updated[-1]

            # REVIEW
            ### gas phase ###
            # check BC
            if z == 0 and solverMeshSet is True:
                # BC1
                BC1_T_1 = PeNuHe0*dz
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC1)
            elif z == 0 and solverMeshSet is False:
                # BC1
                BC1_T_1 = PeNuHe0*dzs[z]
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dzs[z], DIFF1_T_SET)
                # d2Fdz2
                # d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF_T_SET_BC1)
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = 0
                _zR_c = dzs[z]/dzs[z-1]
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
            elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                # NOTE
                # dense section
                # i=2,...,zNoNoDense-1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_bb, T_b, T_c, T_f, T_ff]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = dzs[z-2]/dzs[z-1]
                _zR_c = dzs[z]/dzs[z-1]
                #
                dTdz = FiDiNonUniformDerivative1(
                    dFdz_T, dzs[z], DIFF1_T_SET, _zR_b)
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
            elif z == zNo - 1:
                # BC2
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # forward
                T_f = T_b
                T_ff = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC2)
            else:
                # interior nodes
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2] if z < zNo-2 else 0
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_G)

            # REVIEW
            # cal differentiate
            # backward difference
            # dTdz = (T_c - T_b)/(1*dz)
            # convective term
            _convectiveTerm = -1*v_z_DiLeVa*GaDe_DiLeVa*GaCpMeanMix_DiLeVa*dTdz
            # central difference
            # d2Tdz2 = (T_b - 2*T_c + T_f)/(dz**2)
            # dispersion flux [kJ/m^3.s]
            # _dispersionFluxT = (GaThCoEff*d2Tdz2)*1e-3
            _dispersionFluxT = ((1/PeNuHe0)*GaThCoEff_DiLeVa*d2Tdz2)*1
            # temperature in the catalyst surface [K]
            # Ts_cat
            # outward flux [kJ/m^2.s]
            _inwardFluxT = HeTrCo*SpSuAr*Tf*(Ts_r_cat_gas - T_c)*1e-3
            # total heat transfer between gas and solid [kJ/m^3.s]
            _heTrBeGaSoTerm = (1/GaHeCoTe0)*_inwardFluxT
            # heat exchange term [kJ/m^3.s] -> [no unit]
            _heatExchangeTerm = (1/GaHeCoTe0)*Qm
            # convective flux, diffusive flux, enthalpy of reaction, cooling heat
            # const
            _const1 = GaDe_DiLeVa*GaCpMeanMix_DiLeVa*BeVoFr*(zf/vf)
            _const2 = 1/_const1
            #
            dxdt_T = _const2*(_convectiveTerm + _dispersionFluxT +
                              _heTrBeGaSoTerm + _heatExchangeTerm)
            dxdtMat[indexT][0][z] = dxdt_T

        # NOTE
        # flat
        dxdt = dxdtMat.flatten().tolist()

        # print
        print(f"time: {t} seconds")

        return dxdt

# NOTE
# static heterogenous modeling

    def runM9(self, initGuess=[]):
        """
        modeling case (dimensionless)
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static), Cci, Tc (static, for catalyst)
            CT, GaDe = f(P, T, n)
        numerical method: finite difference
        args:
            initGuess: initial guess from 
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverRootSet = solverConfig['root']
        solverIVPSet = solverConfig['ivp']
        solverMesh = solverConfig['mesh']
        solverMeshSet = True if solverMesh == "normal" else False

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']
        # operation time [s]
        opT = self.modelInput['operating-conditions']['period']
        # process-type
        processType = self.modelInput['operating-conditions']['process-type']

        # reaction list
        reactionDict = self.modelInput['reactions']
        reactionList = rmtUtil.buildReactionList(reactionDict)
        # number of reactions
        reactionListNo = len(reactionList)

        # component list
        compList = self.modelInput['feed']['components']['shell']

        # graph label setting
        # labelList = compList.copy()
        # labelList.append("Temperature")
        # labelList.append("Pressure")
        labelList = pltc.makeLabels(
            compList, ["Gas Temp"], compList, ["Solid Temp"])

        # component no
        compNo = len(compList)
        indexTemp = compNo
        indexPressure = indexTemp + 1
        indexVelocity = indexPressure + 1
        # label id
        labelIndex_ConcGasPhase = 0
        labelIndex_TempGasPhase = compNo
        labelIndex_ConcSolidPhase = labelIndex_TempGasPhase + 1
        labelIndex_TempSolidPhase = labelIndex_ConcSolidPhase + compNo

        # reactor spec
        ReSpec = self.modelInput['reactor']
        # reactor inner diameter [m]
        ReInDi = ReSpec['ReInDi']
        # reactor length [m]
        ReLe = ReSpec['ReLe']
        # cross-sectional area [m^2]
        CrSeAr = CONST.PI_CONST*(ReInDi ** 2)/4
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']

        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [kmol/m^3]
        SpCo0 = np.sum(SpCoi0)
        # inlet superficial velocity [m/s]
        SuGaVe0 = self.modelInput['feed']['superficial-velocity']
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # diffusivity coefficient - gas phase [m^2/s]
        GaDii0 = self.modelInput['feed']['diffusivity']
        # gas viscosity [Pa.s]
        GaVii0 = self.modelInput['feed']['viscosity']
        # gas mixture viscosity [Pa.s]
        GaViMix0 = self.modelInput['feed']['mixture-viscosity']

        # thermal conductivity - gas phase [J/s.m.K]
        GaThCoi0 = self.modelInput['feed']['thermal-conductivity']
        # mixture thermal conductivity - gas phase [J/s.m.K]
        GaThCoMix0 = self.modelInput['feed']['mixture-thermal-conductivity']

        # REVIEW
        # domain length
        DoLe = 1
        # ramp list
        rampList = solverSetting['M9']['rampList']
        # orthogonal collocation points in the r direction
        rNo = solverSetting['M9']['rNo']
        # mesh setting
        zMesh = solverSetting['T1']['zMesh']
        # number of nodes
        zNoNo = zMesh['zNoNo']
        # domain length section
        DoLeSe = zMesh['DoLeSe']
        # mesh refinement degree
        MeReDe = zMesh['MeReDe']
        # mesh installment
        if solverMeshSet is False:
            zMeshRes = FiDiMeshGenerator(zNoNo, DoLe, DoLeSe, MeReDe)
            # finite difference points
            dataXs = zMeshRes['data1']
            # dz lengths
            dzs = zMeshRes['data2']
            # finite difference point number
            zNo = zMeshRes['data3']
            # R ratio
            zR = zMeshRes['data4']
            # dz
            dz = zMeshRes['data5']
        else:
            # finite difference points in the z direction
            zNo = solverSetting['M9']['zNo']
            # length list [reactor length]
            dataXs = np.linspace(0, DoLe, zNo)
            # element size - dz [m]
            dz = DoLe/(zNo-1)
            # reset
            dzs = []
            zR = []

        ### calculation ###
        # mole fraction in the gas phase
        MoFri0 = np.array(rmtUtil.moleFractionFromConcentrationSpecies(SpCoi0))

        # mixture molecular weight [kg/mol]
        MiMoWe0 = rmtUtil.mixtureMolecularWeight(MoFri0, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe0 = calDensityIG(MiMoWe0, SpCo0*1000)

        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
        # Cp mean list
        GaCpMeanList0 = calMeanHeatCapacityAtConstantPressure(compList, T)
        # Cp mixture
        GaCpMeanMix0 = calMixtureHeatCapacityAtConstantPressure(
            MoFri0, GaCpMeanList0)

        # thermal diffusivity in the gas phase [m^2/s]
        GaThDi = calThermalDiffusivity(
            GaThCoMix0, GaDe0, GaCpMeanMix0, MiMoWe0)

        # var no (Ci,T)
        varNo = compNo + 1
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total number
        varNoConInSolid = varNoConInSolidBlock*zNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of layers
        noLayer = compNo+1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = rNo+1
        # number of var columns [i]
        varNoColumns = zNo

        # parameters
        # component data
        reactionListSorted = self.reactionListSorted
        # reaction coefficient
        reactionStochCoeff = self.reactionStochCoeffList

        # standard heat of reaction at 25C [kJ/kmol]
        StHeRe25 = np.array(
            list(map(calStandardEnthalpyOfReaction, reactionList)))

        # REVIEW
        # solver setting
        # orthogonal collocation method
        OrCoClassSet = OrCoClass()
        OrCoClassSetRes = OrCoClassSet.buildMatrix()

        # NOTE
        ### dimensionless analysis ###
        # concentration [kmol/m^3]
        Cif = np.copy(SpCoi0)
        # total concentration
        Cf = SpCo0
        # temperature [K]
        Tf = T
        # superficial velocity [m/s]
        vf = SuGaVe0
        # length [m]
        zf = ReLe
        # diffusivity [m^2/s]
        Dif = np.copy(GaDii0)
        # heat capacity at constant pressure [J/mol.K] | [kJ/kmol.K]
        Cpif = np.copy(GaCpMeanList0)
        # mixture heat capacity [J/mol.K] | [kJ/kmol.K]
        Cpf = GaCpMeanMix0
        # radius
        rf = PaDi/2

        # gas phase
        # mass convective term - (list) [kmol/m^3.s]
        _Cif = Cif if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.repeat(
            np.max(Cif), compNo)
        GaMaCoTe0 = (vf/zf)*_Cif
        # mass diffusive term - (list)  [kmol/m^3.s]
        GaMaDiTe0 = (1/zf**2)*(_Cif*Dif)
        # heat convective term [kJ/m^3.s]
        GaHeCoTe0 = (GaDe0*vf*Tf*(Cpf/MiMoWe0)/zf)*1e-3
        # heat diffusive term [kJ/m^3.s]
        GaHeDiTe0 = (Tf*GaThCoMix0/zf**2)*1e-3

        # solid phase
        # mass diffusive term - (list)  [kmol/m^3.s]
        SoMaDiTe0 = (Dif*_Cif)/rf**2
        # heat diffusive term [kJ/m^3.s]
        SoHeDiTe0 = (GaThCoMix0*Tf/rf**2)*1e-3

        ### dimensionless numbers ###
        # Re Number
        ReNu0 = calReNoEq1(GaDe0, SuGaVe0, PaDi, GaViMix0)
        # Sc Number
        ScNu0 = calScNoEq1(GaDe0, GaViMix0, GaDii0)
        # Sh Number (choose method)
        ShNu0 = calShNoEq1(ScNu0, ReNu0, CONST_EQ_Sh['Frossling'])
        # Prandtl Number
        PrNu0 = calPrNoEq1(GaCpMeanMix0, GaViMix0, GaThCoMix0, MiMoWe0)
        # Nu number
        NuNu0 = calNuNoEq1(PrNu0, ReNu0)
        # Strouhal number
        StNu = 1
        # Peclet number - mass transfer
        PeNuMa0 = (vf*zf)/Dif
        # Peclet number - heat transfer
        PeNuHe0 = (zf*GaDe0*(Cpf/MiMoWe0)*vf)/GaThCoMix0

        ### transfer coefficient ###
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = calMassTransferCoefficientEq1(ShNu0, GaDii0, PaDi)
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = calHeatTransferCoefficientEq1(NuNu0, GaThCoMix0, PaDi)

        # fun parameters
        FunParam = {
            "compList": compList,
            "const": {
                "CrSeAr": CrSeAr,
                "MoWei": MoWei,
                "StHeRe25": StHeRe25,
                "GaMiVi": GaViMix0,
                "varNo": varNo,
                "varNoT": varNoT,
                "reactionListNo": reactionListNo,
            },
            "ReSpec": ReSpec,
            "ExHe": ExHe,
            "constBC1": {
                "VoFlRa0": VoFlRa0,
                "SpCoi0": SpCoi0,
                "SpCo0": SpCo0,
                "P0": P,
                "T0": T,
                "SuGaVe0": SuGaVe0,
                "GaDii0": GaDii0,
                "GaThCoi0": GaThCoi0,
                "GaVii0": GaVii0,
                "GaDe0": GaDe0,
                "GaCpMeanMix0": GaCpMeanMix0,
                "GaThCoMix0": GaThCoMix0
            },
            "meshSetting": {
                "solverMesh": solverMesh,
                "solverMeshSet": solverMeshSet,
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz,
                "dzs": dzs,
                "zR": zR,
                "zNoNo": zNoNo
            },
            "solverSetting": {
                "dFdz": solverSetting['T1']['dFdz'],
                "d2Fdz2": solverSetting['T1']['d2Fdz2'],
                "dTdz": solverSetting['T1']['dTdz'],
                "d2Tdz2": solverSetting['T1']['d2Tdz2'],
                "OrCoClassSetRes": OrCoClassSetRes,
            },
            "reactionRateExpr": reactionRateExpr

        }

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "zf": zf,
            "Dif": Dif,
            "Cpif": Cpif,
            "Cpf": Cpf,
            "rf": rf,
            "GaMaCoTe0": GaMaCoTe0,
            "GaMaDiTe0": GaMaDiTe0,
            "GaHeCoTe0": GaHeCoTe0,
            "GaHeDiTe0": GaHeDiTe0,
            "ReNu0": ReNu0,
            "ScNu0": ScNu0,
            "ShNu0": ShNu0,
            "PrNu0": PrNu0,
            "PeNuMa0": PeNuMa0,
            "PeNuHe0": PeNuHe0,
            "MaTrCo": MaTrCo,
            "HeTrCo": HeTrCo,
            "SoMaDiTe0": SoMaDiTe0,
            "SoHeDiTe0": SoHeDiTe0
        }

        # NOTE
        # initial guess set
        _initGuessVal = initGuess['dataYs']
        _initGuessVal_Concentration = _initGuessVal[:-1]
        _initGuessVal_Temperature = _initGuessVal[-1]
        initGuessConc_DiLeVa = rmtUtil.calDiLessValue(
            _initGuessVal[:-1], np.max(_Cif))
        initGuessTemp_DiLeVa = rmtUtil.calDiLessValue(
            _initGuessVal[-1], Tf, "TEMP")

        # initial guess at t>0 and z>>0
        IVMatrixShape = (noLayer, varNoRows, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # bounds
        BMatrixShape = (noLayer, varNoRows, varNoColumns)
        BUp2D = np.zeros(BMatrixShape)
        BLower2D = np.zeros(BMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayer - 1):
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # gas phase
                        # 0.5  # SpCoi0[m]/np.max(SpCoi0)
                        IV2D[m][j][i] = initGuessConc_DiLeVa[m, i]
                        # set bounds
                        BUp2D[m][j][i] = 1
                        BLower2D[m][j][i] = 0
                    else:
                        # solid phase
                        # 0.5  # SpCoi0[m]/np.max(SpCoi0)
                        IV2D[m][j][i] = initGuessConc_DiLeVa[m, i]
                        # set bounds
                        BUp2D[m][j][i] = 1
                        BLower2D[m][j][i] = 0

        # temperature
        for i in range(varNoColumns):
            for j in range(varNoRows):
                # separate phase
                if j == 0:
                    # gas phase
                    # 0 + 1e-2*varNoColumns  # T
                    IV2D[noLayer - 1][j][i] = initGuessTemp_DiLeVa[i]
                    # set bounds
                    BUp2D[noLayer - 1][j][i] = 1
                    BLower2D[noLayer - 1][j][i] = -1
                else:
                    # solid phase
                    # 0 + 0.99e-2*varNoColumns  # T
                    IV2D[noLayer - 1][j][i] = initGuessTemp_DiLeVa[i]
                    # set bounds
                    BUp2D[noLayer - 1][j][i] = 1
                    BLower2D[noLayer - 1][j][i] = -1

        # flatten IV
        IV = IV2D.flatten()
        BUp = BUp2D.flatten()
        BLower = BLower2D.flatten()

        # set bound
        setBounds = (BLower, BUp)

        # result
        dataPack = []

        # solver setting
        funSet = PackedBedReactorClass.modelEquationM9
        paramsSet = (reactionListSorted, reactionStochCoeff,
                     FunParam, DimensionlessAnalysisParams, processType)

        # NOTE
        rampListLen = len(rampList)
        # ramp nonlinear term
        for k in range(rampListLen):
            rampSet = rampList[k]
            print("rampSet: ", rampSet)
            ### solve a system of nonlinear algebraic equation ###
            if solverRootSet == "fsolve":
                sol = optimize.fsolve(funSet, IV, args=(paramsSet, rampSet))
                # result
                successStatus = True if len(sol) > 0 else False
                # all results
                # components, temperature layers
                dataYs = sol
            elif solverRootSet == "root":
                # root
                # lm, krylov, anderson, hybr, broyden1, linearmixing, diagbroyden, excitingmixing
                sol = optimize.root(funSet, IV, args=(
                    paramsSet, rampSet), method='lm')
                # result
                successStatus = sol.success
                # all results
                # components, temperature layers
                dataYs = sol.x
            elif solverRootSet == "least_squares":
                sol = optimize.least_squares(
                    funSet, IV, bounds=setBounds, args=(paramsSet, rampSet))
                # result
                successStatus = sol.success
                # all results
                # components, temperature layers
                dataYs = sol.x

            # NOTE
            # update initial guess
            IV = dataYs

        # check
        if successStatus is False:
            raise
        else:
            # std format
            dataYs_Reshaped = np.reshape(
                dataYs, (noLayer, varNoRows, varNoColumns))

            # -> concentration
            dataYs_Concentration_DiLeVa = dataYs_Reshaped[:-1]
            # gas phase
            _ConcGasPhase_DiLeVa = dataYs_Concentration_DiLeVa[:, 0, :]
            # solid phase
            _ConcSolidPhase_DiLeVa = dataYs_Concentration_DiLeVa[:, 1:, :]

            # -> temperature
            dataYs_Temperature_DiLeVa = dataYs_Reshaped[-1] if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
                [[0], [0]], zNo).reshape((varNoRows, varNoColumns))
            # gas phase
            _TempGasPhase_DiLeVa = dataYs_Temperature_DiLeVa[0, :].reshape(
                (1, zNo))
            # solid phase
            _TempSolidPhase_DiLeVa = dataYs_Temperature_DiLeVa[1:, :]

            # sort out
            params1 = (compNo, noLayer, varNoRows, varNoColumns, rNo, zNo)
            params2 = (Cif, Tf, processType)
            dataYs_Sorted = sortedResult3(
                _ConcGasPhase_DiLeVa, _TempGasPhase_DiLeVa, _ConcSolidPhase_DiLeVa, _TempSolidPhase_DiLeVa, params1, params2)
            # gas phase
            # component concentration [kmol/m^3]
            _ConcGasPhase_ReVa = dataYs_Sorted['data1']
            # temperature [K]
            _TempGasPhase_ReVa = dataYs_Sorted['data2']
            # solid phase
            # component concentration [kmol/m^3]
            _ConcSolidPhase_ReVa = dataYs_Sorted['data3']
            # temperature [K]
            _TempSolidPhase_ReVa = dataYs_Sorted['data4']

            # REVIEW
            # convert concentration to mole fraction
            _ConcGasPhaseTot = np.sum(_ConcGasPhase_ReVa, axis=0)
            _MoFriGasPhase = _ConcGasPhase_ReVa/_ConcGasPhaseTot
            # convert concentration to mole fraction
            _ConcSolidPhaseTot = np.sum(_ConcSolidPhase_ReVa, axis=0)
            _MoFriSolidPhase = _ConcSolidPhase_ReVa/_ConcSolidPhaseTot

            # combine
            # gas phase
            _dataYs_GasPhase = np.concatenate(
                (_MoFriGasPhase, _TempGasPhase_ReVa), axis=0)
            # solid phase
            _MoFriSolidPhase_Reshaped = np.zeros((compNo, varNoColumns))
            for j in range(compNo):
                _MoFriSolidPhase_Reshaped[j, :] = _MoFriSolidPhase[j]
            # set
            _dataYs_SolidPhase = np.concatenate(
                (_MoFriSolidPhase_Reshaped, _TempSolidPhase_ReVa), axis=0)
            #
            _dataYs = np.concatenate(
                (_dataYs_GasPhase, _dataYs_SolidPhase), axis=0)

            # save data
            # dataPack.append({
            #     "successStatus": successStatus,
            #     "dataYCon": dataYs1GasPhase,
            #     "dataYTemp": dataYs2GasPhase,
            #     "dataYs": _dataYs,
            #     "dataYCons": dataYs1SolidPhase,
            #     "dataYTemps": dataYs2SolidPhase,
            # })

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # plot info
        plotTitle = f"Steady-State {processType} Modeling [M14] finished in {elapsed} seconds"
        xLabelSet = "Dimensionless Particle Radius"
        yLabelSet = "Dimensionless Concentration"

        # REVIEW
        # *** post-processing results ***
        # plot setting: build (x,y) series
        XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
        # -> add label
        dataList = pltc.plots2DSetDataList(XYList, labelList)
        # datalists
        dataLists = [dataList[0:compNo], dataList[indexTemp],
                     dataList[labelIndex_ConcSolidPhase:labelIndex_ConcSolidPhase+compNo], dataList[labelIndex_TempSolidPhase]]
        # subplot result
        pltc.plots2DSub(dataLists, xLabelSet, yLabelSet, plotTitle)

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationM9(y, paramsSet, rampSet=1):
        """
            model [static modeling]
            mass, energy, and momentum balance equations
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
                        varNo: number of variables (Ci, CT, T)
                        varNoT: number of variables in the domain (zNo*varNoT)
                        reactionListNo: reaction list number
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1:
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [kmol/m^3],
                        SpCo0: total concentration [kmol/m^3]
                        P0: inlet pressure [Pa]
                        T0: inlet temperature [K]
                    meshSetting:
                        solverMesh: mesh installment
                        solverMeshSet: 
                            true: normal
                            false: mesh refinement
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                        dzs: differential length list [-]
                        zR: z ratio
                        zNoNo: number of nodes in the dense and normal sections
                    solverSetting:
                        OrCoClassSetRes: constants of OC methods
                    reactionRateExpr: reaction rate expressions []
                DimensionlessAnalysisParams:
                    Cif: feed concentration [kmol/m^3]
                    Tf: feed temperature
                    vf: feed superficial velocity [m/s]
                    zf: domain length [m]
                    Dif: diffusivity coefficient of component [m^2/s]
                    Cpif: feed heat capacity at constat pressure [kJ/kmol.K] | [J/mol.K]
                    rf: particle radius [m]
                    GaMaCoTe0: feed mass convective term of gas phase [kmol/m^3.s]
                    GaMaDiTe0: feed mass diffusive term of gas phase [kmol/m^3.s]
                    GaHeCoTe0: feed heat convective term of gas phase [kJ/m^3.s]
                    GaHeDiTe0, feed heat diffusive term of gas phase [kJ/m^3.s]
                    SoMaDiTe0: feed mass diffusive term of solid phase [kmol/m^3.s]
                    SoHeDiTe0: feed heat diffusive term of solid phase [kJ/m^3.s]
                    ReNu0: Reynolds number
                    ScNu0: Schmidt number
                    ShNu0: Sherwood number
                    PrNu0: Prandtl number
                    PeNuMa0: mass Peclet number
                    PeNuHe0: heat Peclet number 
                    MaTrCo: mass transfer coefficient - gas/solid [m/s]
                    HeTrCo: heat transfer coefficient - gas/solid [J/m^2.s.K]
                processType
        """
        # parameters
        reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType = paramsSet
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

        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']
        # catalyst density [kgcat/m^3 of particle]
        CaDe = ReSpec['CaDe']
        # catalyst heat capacity at constant pressure [kJ/kg.K]
        CaSpHeCa = ReSpec['CaSpHeCa']
        # catalyst porosity
        CaPo = ReSpec['CaPo']
        # catalyst tortuosity
        CaTo = ReSpec['CaTo']
        # catalyst thermal conductivity [J/K.m.s]
        CaThCo = ReSpec['CaThCo']

        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # boundary conditions constants
        constBC1 = FunParam['constBC1']
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = constBC1['VoFlRa0']
        # inlet species concentration [kmol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [kmol/m^3]
        SpCo0 = constBC1['SpCo0']
        # inlet pressure [Pa]
        P0 = constBC1['P0']
        # inlet temperature [K]
        T0 = constBC1['T0']
        # inlet superficial velocity [m/s]
        # SuGaVe0 = constBC1['SuGaVe0']
        # inlet diffusivity coefficient [m^2]
        GaDii0 = constBC1['GaDii0']
        # inlet gas thermal conductivity [J/s.m.K]
        GaThCoi0 = constBC1['GaThCoi0']
        # gas viscosity
        GaVii0 = constBC1['GaVii0']
        # gas density [kg/m^3]
        GaDe0 = constBC1['GaDe0']
        # heat capacity at constant pressure [kJ/kmol.K] | [J/mol.K]
        GaCpMeanMix0 = constBC1['GaCpMeanMix0']
        # gas thermal conductivity [J/s.m.K]
        GaThCoMix0 = constBC1['GaThCoMix0']

        # mesh setting
        meshSetting = FunParam['meshSetting']
        # mesh installment
        solverMesh = meshSetting['solverMesh']
        # mesh refinement
        solverMeshSet = meshSetting['solverMeshSet']
        # number of layers
        noLayer = meshSetting['noLayer']
        # var no in each layer
        varNoLayer = meshSetting['varNoLayer']
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = meshSetting['varNoLayerT']
        # number of var rows [j]
        varNoRows = meshSetting['varNoRows']
        # number of var columns [i]
        varNoColumns = meshSetting['varNoColumns']
        # rNo
        rNo = meshSetting['rNo']
        # zNo
        zNo = meshSetting['zNo']
        # dz [m]
        dz = meshSetting['dz']
        # dzs [m]/[-]
        dzs = meshSetting['dzs']
        # R ratio
        zR = meshSetting['zR']
        # number of nodes in the dense and normal sections
        zNoNo = meshSetting['zNoNo']
        # dense
        zNoNoDense = zNoNo[0]
        # normal
        zNoNoNormal = zNoNo[1]

        # solver setting
        solverSetting = FunParam['solverSetting']
        # mass balance equation
        DIFF1_C_SET = solverSetting['dFdz']
        DIFF2_C_SET_BC1 = solverSetting['d2Fdz2']['BC1']
        DIFF2_C_SET_BC2 = solverSetting['d2Fdz2']['BC2']
        DIFF2_C_SET_G = solverSetting['d2Fdz2']['G']
        # energy balance equation
        DIFF1_T_SET = solverSetting['dTdz']
        DIFF2_T_SET_BC1 = solverSetting['d2Tdz2']['BC1']
        DIFF2_T_SET_BC2 = solverSetting['d2Tdz2']['BC2']
        DIFF2_T_SET_G = solverSetting['d2Tdz2']['G']
        # number of collocation points
        ocN = solverSetting['OrCoClassSetRes']['N']
        ocXc = solverSetting['OrCoClassSetRes']['Xc']
        ocA = solverSetting['OrCoClassSetRes']['A']
        ocB = solverSetting['OrCoClassSetRes']['B']
        ocQ = solverSetting['OrCoClassSetRes']['Q']

        # init OrCoCatParticle
        OrCoCatParticleClassSet = OrCoCatParticleClass(
            ocXc, ocN, ocQ, ocA, ocB, varNo)

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # dimensionless analysis params

        #  feed concentration [kmol/m^3]
        Cif = DimensionlessAnalysisParams['Cif']
        # feed temperature
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # domain length [m]
        zf = DimensionlessAnalysisParams['zf']
        # particle radius [m]
        rf = DimensionlessAnalysisParams['rf']
        # diffusivity coefficient of component [m^2/s]
        Dif = DimensionlessAnalysisParams['Dif']
        # feed heat capacity at constat pressure
        Cpif = DimensionlessAnalysisParams['Cpif']
        # feed mass convective term of gas phase [kmol/m^3.s]
        GaMaCoTe0 = DimensionlessAnalysisParams['GaMaCoTe0']
        # feed mass diffusive term of gas phase [kmol/m^3.s]
        GaMaDiTe0 = DimensionlessAnalysisParams['GaMaDiTe0']
        # feed heat convective term of gas phase [kJ/m^3.s]
        GaHeCoTe0 = DimensionlessAnalysisParams['GaHeCoTe0']
        # feed heat diffusive term of gas phase [kJ/m^3.s]
        GaHeDiTe0 = DimensionlessAnalysisParams['GaHeDiTe0']
        # feed mass diffusive term of solid phase [kmol/m^3.s]
        SoMaDiTe0 = DimensionlessAnalysisParams['SoMaDiTe0']
        # feed heat diffusive term of solid phase [kJ/m^3.s]
        SoHeDiTe0 = DimensionlessAnalysisParams['SoHeDiTe0']
        # Reynolds number
        ReNu = DimensionlessAnalysisParams['ReNu0']
        # Schmidt number
        ScNu = DimensionlessAnalysisParams['ScNu0']
        # Sherwood number
        ShNu = DimensionlessAnalysisParams['ShNu0']
        # Prandtl number
        PrNu = DimensionlessAnalysisParams['PrNu0']
        # mass Peclet number
        PeNuMa0 = DimensionlessAnalysisParams['PeNuMa0']
        # heat Peclet number
        PeNuHe0 = DimensionlessAnalysisParams['PeNuHe0']
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = DimensionlessAnalysisParams['MaTrCo']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = DimensionlessAnalysisParams['HeTrCo']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo
        indexP = indexT + 1
        indexV = indexP + 1

        # calculate
        # particle radius
        PaRa = PaDi/2
        # specific surface area exposed to the free fluid [m^2/m^3]
        SpSuAr = (3/PaRa)*(1 - BeVoFr)

        # molar flowrate [kmol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        InGaVe0 = VoFlRa0/(CrSeAr*BeVoFr)
        # interstitial gas velocity [m/s]
        SuGaVe0 = InGaVe0*BeVoFr

        # interstitial gas velocity [m/s]
        InGaVeList_z = np.zeros(zNo)
        InGaVeList_z[0] = InGaVe0

        # total molar flux [kmol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate in the solid phase
        Ri_z = np.zeros((zNo, reactionListNo))
        Ri_zr = np.zeros((zNo, rNo, reactionListNo))
        Ri_r = np.zeros((rNo, reactionListNo))
        # reaction rate
        # ri = np.zeros(compNo) # deprecate
        # ri0 = np.zeros(compNo) # deprecate
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        # overall reaction
        OvR = np.zeros(rNo)
        # overall enthalpy
        OvHeReT = np.zeros(rNo)
        # heat capacity at constant pressure
        SoCpMeanMix = np.zeros(rNo)
        # effective heat capacity at constant pressure
        SoCpMeanMixEff = np.zeros(rNo)
        # dimensionless analysis
        SoCpMeanMixEff_ReVa = np.zeros(rNo)

        # pressure [Pa]
        P_z = np.zeros(zNo + 1)
        P_z[0] = P0

        # superficial gas velocity [m/s]
        v_z = np.zeros(zNo + 1)
        v_z[0] = SuGaVe0

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (noLayer, varNoRows, varNoColumns))

        # all species concentration in gas & solid phase
        SpCo_mz = np.zeros((noLayer - 1, varNoRows, varNoColumns))
        # all species concentration in gas phase [kmol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
        # all species concentration in solid phase (catalyst) [kmol/m^3]
        SpCosi_mzr = np.zeros((compNo, rNo, zNo))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCo_mz[m] = _SpCoi
        # concentration in the gas phase [kmol/m^3]
        for m in range(compNo):
            for j in range(varNoRows):
                if j == 0:
                    # gas phase
                    SpCoi_z[m, :] = SpCo_mz[m, j, :]
                else:
                    # solid phase
                    SpCosi_mzr[m, j-1, :] = SpCo_mz[m, j, :]

        # species concentration in gas phase [kmol/m^3]
        CoSpi = np.zeros(compNo)
        # dimensionless analysis
        CoSpi_ReVa = np.zeros(compNo)
        # total concentration [kmol/m^3]
        CoSp = 0
        # species concentration in solid phase (catalyst) [kmol/m^3]
        # shape
        CosSpiMatShape = (rNo, compNo)
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)
        # total concentration in the solid phase [kmol/m^3]
        CosSp_r = np.zeros(rNo)

        # flux
        MoFli_z = np.zeros(compNo)

        # NOTE
        # temperature [K]
        T_mz = np.zeros((varNoRows, varNoColumns))
        T_mz = yLoop[noLayer -
                     1] if processType != "iso-thermal" else np.repeat([[0], [0]], zNo).reshape((varNoRows, varNoColumns))
        # temperature in the gas phase
        T_z = np.zeros(zNo)
        T_z = T_mz[0, :]
        # temperature in solid phase
        Ts_z = np.zeros((rNo, zNo))
        Ts_z = T_mz[1:]
        # temperature in the solid phase
        Ts_r = np.zeros(rNo)

        # diff/dt
        # dxdt = []
        # matrix
        # dxdtMat = np.zeros((varNo, zNo))
        dxdtMat = np.zeros((noLayer, varNoRows, varNoColumns))

        # NOTE
        # FIXME
        # define ode equations for each finite difference [zNo]
        for z in range(varNoColumns):
            ## block ##
            # concentration species in the gas phase [kmol/m^3]
            for i in range(compNo):
                _SpCoi_z = SpCoi_z[i][z]
                CoSpi[i] = _SpCoi_z  # max(_SpCoi_z, CONST.EPS_CONST)
                # REVIEW
                # dimensionless analysis: real value
                SpCoi0_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                CoSpi_ReVa[i] = rmtUtil.calRealDiLessValue(
                    CoSpi[i], SpCoi0_Set)

            # total concentration [kmol/m^3]
            CoSp = np.sum(CoSpi)
            # dimensionless analysis: real value
            CoSp_ReVa = np.sum(CoSpi_ReVa)

            # FIXME
            # concentration species in the solid phase [kmol/m^3]
            # display concentration list in each oc point (rNo)
            for i in range(compNo):
                for r in range(rNo):
                    _CosSpi_z = SpCosi_mzr[i][r][z]
                    # max(_CosSpi_z, CONST.EPS_CONST)
                    CosSpi_r[r][i] = _CosSpi_z
                    # REVIEW
                    # dimensionless analysis: real value
                    SpCoi0_r_Set = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                        SpCoi0)
                    CosSpi_r_ReVa[r][i] = rmtUtil.calRealDiLessValue(
                        CosSpi_r[r][i], SpCoi0_r_Set)

            # total concentration in the solid phase [kmol/m^3]
            CosSp_r = np.sum(CosSpi_r, axis=1).reshape((rNo, 1))
            # dimensionless analysis: real value
            CosSp_r_ReVa = np.sum(CosSpi_r_ReVa, axis=1).reshape((rNo, 1))

            # concentration in the outer surface of the catalyst [kmol/m^3]
            CosSpi_cat = CosSpi_r[0]
            # dimensionless analysis
            CosSpi_cat_DiLeVa = CosSpi_r[0, :]

            # temperature [K]
            T = T_z[z]
            T_ReVa = rmtUtil.calRealDiLessValue(T, Tf, "TEMP")
            # temperature in the solid phase (for each point)
            # Ts[3], Ts[2], Ts[1], Ts[0]
            Ts_r = Ts_z[:, z]
            Ts_r_ReVa = rmtUtil.calRealDiLessValue(Ts_r, Tf, "TEMP")
            # REVIEW
            print("z: ", z, "T_ReVa: ", T_ReVa,  " Ts_r_ReVa: ", Ts_r_ReVa)

            # pressure [Pa]
            P = P_z[z]

            # FIXME
            # velocity
            # dimensionless value
            # v = v_z[z]
            v = 1

            ## calculate ##
            # mole fraction in the gas phase
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi_ReVa))

            # mole fraction in the solid phase
            # MoFrsi_r0 = CosSpi_r/CosSp_r
            MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
                CosSpi_r_ReVa)

            # TODO
            # dv/dz
            # gas velocity based on interstitial velocity [m/s]
            # InGaVe = rmtUtil.calGaVeFromEOS(InGaVe0, SpCo0, CoSp, P0, P)
            # superficial gas velocity [m/s]
            # SuGaVe = InGaVe*BeVoFr
            # from ode eq. dv/dz
            SuGaVe = v
            # dimensionless analysis
            SuGaVe_ReVa = rmtUtil.calRealDiLessValue(SuGaVe, SuGaVe0)

            # total flowrate [kmol/s]
            # [kmol/m^3]*[m/s]*[m^2]
            MoFlRa = calMolarFlowRate(CoSp_ReVa, SuGaVe_ReVa, CrSeAr)
            # molar flowrate list [kmol/s]
            MoFlRai = MoFlRa*MoFri
            # convert to [mol/s]
            MoFlRai_Con1 = 1000*MoFlRai

            # molar flux [kmol/m^2.s]
            MoFl = MoFlRa/CrSeAr

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai_Con1)

            # mixture molecular weight [kg/mol]
            MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

            # gas density [kg/m^3]
            GaDe = calDensityIG(MiMoWe, CoSp_ReVa*1000)
            # GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)
            # dimensionless value
            GaDe_DiLeVa = rmtUtil.calDiLessValue(GaDe, GaDe0)

            # NOTE
            # ergun equation
            ergA = 150*GaMiVi*SuGaVe_ReVa/(PaDi**2)
            ergB = ((1-BeVoFr)**2)/(BeVoFr**3)
            ergC = 1.75*GaDe*(SuGaVe_ReVa**2)/PaDi
            ergD = (1-BeVoFr)/(BeVoFr**3)
            RHS_ergun = -1*(ergA*ergB + ergC*ergD)

            # momentum balance (ergun equation)
            dxdt_P = RHS_ergun
            # FIXME
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # REVIEW
            # FIXME
            # viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaVii = GaVii0 if MODEL_SETTING['GaVii'] == "FIX" else calTest()
            # mixture viscosity in the gas phase [Pa.s] | [kg/m.s]
            # FIXME
            GaViMix = 2.5e-5  # f(yi,GaVi,MWs);
            # kinematic viscosity in the gas phase [m^2/s]
            GaKiViMix = GaViMix/GaDe

            # REVIEW
            # FIXME
            # solid gas thermal conductivity
            SoThCoMix0 = GaThCoMix0
            # add loop for each r point/constant
            # catalyst thermal conductivity [J/s.m.K]
            # CaThCo
            # membrane wall thermal conductivity [J/s.m.K]
            MeThCo = 1
            # thermal conductivity - gas phase [J/s.m.K]
            # GaThCoi = np.zeros(compNo)  # f(T);
            GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest(
            )
            # dimensionless
            GaThCoi_DiLe = GaThCoi/GaThCoi0
            # FIXME
            # mixture thermal conductivity - gas phase [J/s.m.K]
            GaThCoMix = GaThCoMix0
            # dimensionless analysis
            GaThCoMix_DiLeVa = GaThCoMix/GaThCoMix0
            # thermal conductivity - solid phase [J/s.m.K]
            # assume the same as gas phase
            # SoThCoi = np.zeros(compNo)  # f(T);
            SoThCoi = GaThCoi
            # mixture thermal conductivity - solid phase [J/s.m.K]
            SoThCoMix = GaThCoMix0
            # dimensionless analysis
            SoThCoMix_DiLeVa = SoThCoMix/SoThCoMix0
            # effective thermal conductivity - gas phase [J/s.m.K]
            # GaThCoEff = BeVoFr*GaThCoMix + (1 - BeVoFr)*CaThCo
            GaThCoEff = BeVoFr*GaThCoMix
            # dimensionless analysis
            GaThCoEff_DiLeVa = BeVoFr*GaThCoMix_DiLeVa
            # FIXME
            # effective thermal conductivity - solid phase [J/s.m.K]
            # assume identical to gas phase
            # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
            # SoThCoEff = SoThCoMix*((1 - CaPo)/CaTo)
            SoThCoEff = CaPo*SoThCoMix
            # dimensionless analysis
            # SoThCoEff_DiLeVa = GaThCoMix_DiLeVa*((1 - CaPo)/CaTo)
            SoThCoEff_DiLeVa = CaPo*SoThCoMix_DiLeVa

            # REVIEW
            # diffusivity coefficient - gas phase [m^2/s]
            GaDii = GaDii0 if MODEL_SETTING['GaDii'] == "FIX" else calTest()
            # dimensionless analysis
            GaDii_DiLeVa = GaDii/GaDii0
            # effective diffusivity coefficient - gas phase
            GaDiiEff = GaDii*BeVoFr
            # dimensionless analysis
            GaDiiEff_DiLeVa = GaDiiEff/GaDii0
            # effective diffusivity - solid phase [m^2/s]
            SoDiiEff = (CaPo/CaTo)*GaDii
            # dimensionless analysis
            SoDiiEff_DiLe = (CaPo/CaTo)*GaDii_DiLeVa

            # REVIEW
            if MODEL_SETTING['MaTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Re Number
                ReNu = calReNoEq1(GaDe, SuGaVe, PaDi, GaViMix)
                # Sc Number
                ScNu = calScNoEq1(GaDe, GaViMix, GaDii)
                # Sh Number (choose method)
                ShNu = calShNoEq1(ScNu, ReNu, CONST_EQ_Sh['Frossling'])

                # mass transfer coefficient - gas/solid [m/s]
                MaTrCo = calMassTransferCoefficientEq1(ShNu, GaDii, PaDi)

            # NOTE
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
            for r in range(rNo):
                # loop
                loopVars0 = (Ts_r_ReVa[r], P_z[z],
                             MoFrsi_r[r], CosSpi_r_ReVa[r])

                # component formation rate [mol/m^3.s]
                # check unit
                r0 = np.array(reactionRateExe(
                    loopVars0, varisSet, ratesSet))

                # loop
                Ri_zr[z, r, :] = r0
                Ri_r[r, :] = rampSet*r0

                # REVIEW
                # add a ramp term to improve convergence
                # component formation rate [kmol/m^3.s]
                ri_r[r] = componentFormationRate(
                    compNo, comList, reactionStochCoeff, Ri_r[r])

                # overall formation rate [kmol/m^3.s]
                OvR[r] = np.sum(ri_r[r])

            # NOTE
            ### enthalpy calculation ###
            # gas phase
            # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
            # Cp mean list
            GaCpMeanList = calMeanHeatCapacityAtConstantPressure(
                comList, T_ReVa)
            # Cp mixture
            GaCpMeanMix = calMixtureHeatCapacityAtConstantPressure(
                MoFri, GaCpMeanList)
            # dimensionless analysis
            GaCpMeanMix_DiLeVa = rmtUtil.calDiLessValue(
                GaCpMeanMix, GaCpMeanMix0)
            # effective heat capacity - gas phase [kJ/kmol.K] | [J/mol.K]
            GaCpMeanMixEff = GaCpMeanMix*BeVoFr
            # dimensionless analysis
            GaCpMeanMixEff_DiLeVa = GaCpMeanMix_DiLeVa*BeVoFr

            # solid phase
            for r in range(rNo):
                # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
                # Cp mean list
                SoCpMeanList = calMeanHeatCapacityAtConstantPressure(
                    comList, Ts_r[r])
                # Cp mixture
                SoCpMeanMix[r] = calMixtureHeatCapacityAtConstantPressure(
                    MoFrsi_r[r], SoCpMeanList)

                # effective heat capacity - solid phase [kJ/m^3.K]
                SoCpMeanMixEff_ReVa[r] = CosSp_r_ReVa[r] * \
                    SoCpMeanMix[r]*CaPo + (1-CaPo)*CaDe*CaSpHeCa

                # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
                # enthalpy change
                EnChList = np.array(
                    calEnthalpyChangeOfReaction(reactionListSorted, Ts_r[r]))
                # heat of reaction at T [kJ/kmol] | [J/mol]
                HeReT = np.array(EnChList + StHeRe25)
                # overall heat of reaction [kJ/m^3.s]
                # exothermic reaction (negative sign)
                # endothermic sign (positive sign)
                OvHeReT[r] = np.dot(Ri_r[r, :], HeReT)

            # REVIEW
            if MODEL_SETTING['HeTrCo'] != "FIX":
                ### dimensionless numbers ###
                # Prandtl Number
                # MW kg/mol -> g/mol
                # MiMoWe_Conv = 1000*MiMoWe
                PrNu = calPrNoEq1(
                    GaCpMeanMix, GaViMix, GaThCoMix, MiMoWe)
                # Nu number
                NuNu = calNuNoEq1(PrNu, ReNu)
                # heat transfer coefficient - gas/solid [J/m^2.s.K]
                HeTrCo = calHeatTransferCoefficientEq1(NuNu, GaThCoMix, PaDi)

            # REVIEW
            # heat transfer coefficient - medium side [J/m2.s.K]
            # hs = heat_transfer_coefficient_shell(T,Tv,Pv,Pa);
            # overall heat transfer coefficient [J/m2.s.K]
            # U = overall_heat_transfer_coefficient(hfs,kwall,do,di,L);
            # heat transfer coefficient - permeate side [J/m2.s.K]

            # NOTE
            # cooling temperature [K]
            Tm = ExHe['MeTe']
            # overall heat transfer coefficient [J/s.m2.K]
            U = ExHe['OvHeTrCo']
            # heat transfer area over volume [m^2/m^3]
            a = ExHe['EfHeTrAr']
            # heat transfer parameter [W/m^3.K] | [J/s.m^3.K]
            # Ua = U*a
            # external heat [kJ/m^3.s]
            Qm = rmtUtil.calHeatExchangeBetweenReactorMedium(
                Tm, T_ReVa, U, a, 'kJ/m^3.s')

            # NOTE
            # # mass transfer between
            # for i in range(compNo):
            #     ### gas phase ###
            #     # mass balance (forward difference)
            #     # concentration [kmol/m^3]
            #     # central
            #     Ci_c = SpCoi_z[i][z]
            #     # concentration in the catalyst surface [kmol/m^3]
            #     # CosSpi_cat
            #     # dimensionless analysis: real value
            #     Ci_f = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
            #         SpCoi0)
            #     # inward flux [kmol/m^2.s]
            #     MoFli_z[i] = MaTrCo[i]*Ci_f*(Ci_c - CosSpi_cat_DiLeVa[i])

            # # total mass transfer between gas and solid phases [kmol/m^3]
            # ToMaTrBeGaSo_z = np.sum(MoFli_z)*SpSuAr

            # NOTE
            # velocity from global concentration
            # check BC
            # if z == 0:
            #     # BC1
            #     constT_BC1 = (GaThCoEff)/(MoFl*GaCpMeanMix/1000)
            #     # next node
            #     T_f = T_z[z+1]
            #     # previous node
            #     T_b = (T0*dz + constT_BC1*T_f)/(dz + constT_BC1)
            # elif z == zNo - 1:
            #     # BC2
            #     # previous node
            #     T_b = T_z[z - 1]
            #     # next node
            #     T_f = 0
            # else:
            #     # interior nodes
            #     T_b = T_z[z-1]
            #     # next node
            #     T_f = T_z[z+1]

            # dxdt_v_T = (T_z[z] - T_b)/dz
            # # CoSp x 1000
            # # OvR x 1000
            # dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
            #                           ((1/T_z[z])*dxdt_P - (P_z[z]/T_z[z]**2)*dxdt_v_T) - ToMaTrBeGaSo_z*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            # FIXME
            # v_z[z+1] = dxdt_v*dz + v_z[z]
            # v_z[z+1] = v
            # FIXME
            v_z[z+1] = v_z[z]
            # dimensionless analysis
            v_z_DiLeVa = rmtUtil.calDiLessValue(v_z[z+1], vf)

            # NOTE
            # diff/dt
            # dxdt = []
            # matrix
            # dxdtMat = np.zeros((varNo, zNo))

            # bulk temperature [K]
            T_c = T_z[z]

            # REVIEW
            # gas-solid interface BC
            # concentration [m/s]*[m^2/s]=[1/m]
            # betaC = PaRa*(MaTrCo/SoDiiEff)
            # temperature
            # betaT = -1*((HeTrCo*PaRa)/SoThCoEff)

            # universal index [j,i]
            # UISet = z*(rNo + 1)

            # NOTE
            for i in range(compNo):
                # concentration []
                # bulk species
                Ci_c = SpCoi_z[i][z]
                # species concentration at different points of particle radius [rNo]
                # [Cs[3], Cs[2], Cs[1], Cs[0]]
                _Cs_r = CosSpi_r[:, i].flatten()

                # REVIEW
                ### gas phase ###
                # check BC
                if z == 0 and solverMeshSet is True:
                    # NOTE
                    # BC1 (normal)
                    BC1_C_1 = PeNuMa0[i]*dz
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC1)
                elif z == 0 and solverMeshSet is False:
                    # NOTE
                    # BC1 (dense)
                    # i=0 is discretized based on inlet
                    # i=1
                    BC1_C_1 = PeNuMa0[i]*dzs[z]
                    BC1_C_2 = 1/BC1_C_1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    # GaDii_DiLeVa = 1
                    Ci_0 = 1 if MODEL_SETTING['GaMaCoTe0'] != "MAX" else SpCoi0[i]/np.max(
                        SpCoi0)
                    Ci_b = (Ci_0 + BC1_C_2*Ci_f)/(BC1_C_2 + 1)
                    Ci_bb = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dzs[z], DIFF1_C_SET)
                    # d2Fdz2
                    # d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1)
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = 0
                    _zR_c = dzs[z]/dzs[z-1]
                    # dCdz = FiDiNonUniformDerivative1(
                    #     dFdz_C, dzs[z], DIFF1_C_SET, zR[z])
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_BC1, _zR_c)
                elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                    # NOTE
                    # dense section
                    # i=2,...,zNoNoDense-1
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2]
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### non-uniform nodes ###
                    # R value
                    _zR_b = dzs[z-2]/dzs[z-1]
                    _zR_c = dzs[z]/dzs[z-1]
                    #
                    dCdz = FiDiNonUniformDerivative1(
                        dFdz_C, dzs[z], DIFF1_C_SET, _zR_b)
                    # d2Fdz2
                    d2Cdz2 = FiDiNonUniformDerivative2(
                        d2Fdz2_C, dzs[z], DIFF2_C_SET_G, _zR_c)
                elif z == zNo - 1:
                    # NOTE
                    # BC2
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # forward difference
                    Ci_f = Ci_b
                    Ci_ff = 0
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_BC2)
                else:
                    # NOTE
                    # normal sections
                    # interior nodes
                    # forward
                    Ci_f = SpCoi_z[i][z+1]
                    Ci_ff = SpCoi_z[i][z+2] if z < zNo-2 else 0
                    # backward
                    Ci_b = SpCoi_z[i][z-1]
                    Ci_bb = SpCoi_z[i][z-2]
                    # function value
                    dFdz_C = [Ci_b, Ci_c, Ci_f]
                    d2Fdz2_C = [Ci_bb, Ci_b, Ci_c, Ci_f, Ci_ff]
                    # REVIEW
                    ### uniform nodes ###
                    # dFdz
                    dCdz = FiDiDerivative1(dFdz_C, dz, DIFF1_C_SET)
                    # d2Fdz2
                    d2Cdz2 = FiDiDerivative2(d2Fdz2_C, dz, DIFF2_C_SET_G)

                # REVIEW
                # *** convective flux between fluid-solid ***
                # concentration
                # Ci_c = SpCoi_z[i][z]
                # concentration in the catalyst surface [kmol/m^3]
                CosSpi_cat_gas = _Cs_r[0]
                # dimensionless analysis: real value
                Ci_f = SpCoi0[i] if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.max(
                    SpCoi0)
                # inward flux [kmol/m^2.s]
                MoFli_z[i] = MaTrCo[i]*Ci_f*(Ci_c - CosSpi_cat_gas)

                # REVIEW
                # cal differentiate
                # backward difference
                # dCdz = (Ci_c - Ci_b)/(1*dz)
                # convective term
                _convectiveTerm = -1*v_z_DiLeVa*dCdz
                # central difference for dispersion
                # d2Cdz2 = (Ci_b - 2*Ci_c + Ci_f)/(dz**2)
                # dispersion term [kmol/m^3.s]
                _dispersionFluxC = (BeVoFr*GaDii_DiLeVa[i]/PeNuMa0[i])*d2Cdz2
                # concentration in the catalyst surface [kmol/m^3]
                # CosSpi_cat
                # inward flux [kmol/m^2.s]
                # MoFli_z[i] = MaTrCo[i]*(Ci_c - CosSpi_cat[i])
                _inwardFlux = (1/GaMaCoTe0[i])*MoFli_z[i]*SpSuAr
                # mass balance
                # convective, dispersion, inward flux

                # steady-state
                dxdt_F = _convectiveTerm + _dispersionFluxC - _inwardFlux
                dxdtMat[i][0][z] = dxdt_F

                # REVIEW
                ### solid phase ###
                # transfer from gas to solid surface and then reaction
                # dimensionless analysis
                # beta
                # const
                _alpha = rf/GaDii0[i]
                _beta = MaTrCo[i]/GaDii_DiLeVa[i]
                _DiLe = _alpha*_beta
                _Ri = ri_r[:, i]

                dxdtMat[i][1][z] = MoFli_z[i]*SpSuAr + _Ri

            # NOTE
            # energy balance
            # bulk temperature [K]
            # T_c
            # T_c = T_z[z]
            # REVIEW
            ### solid phase ###
            # temperature at different points of particle radius [rNo]
            # Ts[3], Ts[2], Ts[1], Ts[0]
            _Ts_r = Ts_r.flatten()
            # _Ts_r
            # updated temperature in the gas-solid interface
            Ts_r_cat_gas = _Ts_r[0]

            # REVIEW
            ### gas phase ###
            # check BC
            if z == 0 and solverMeshSet is True:
                # BC1
                BC1_T_1 = PeNuHe0*dz
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC1)
            elif z == 0 and solverMeshSet is False:
                # BC1
                BC1_T_1 = PeNuHe0*dzs[z]
                BC1_T_2 = 1/BC1_T_1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                # GaDe_DiLeVa, GaCpMeanMix_DiLeVa, v_z_DiLeVa = 1
                # T*[0] = (T0 - Tf)/Tf
                T_0 = 0
                T_b = (T_0 + BC1_T_2*T_f)/(BC1_T_2 + 1)
                T_bb = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dzs[z], DIFF1_T_SET)
                # d2Fdz2
                # d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF_T_SET_BC1)
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = 0
                _zR_c = dzs[z]/dzs[z-1]
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
            elif (z > 0 and z < zNoNoDense) and solverMeshSet is False:
                # NOTE
                # dense section
                # i=2,...,zNoNoDense-1
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2]
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_bb, T_b, T_c, T_f, T_ff]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### non-uniform nodes ###
                # R value
                _zR_b = dzs[z-2]/dzs[z-1]
                _zR_c = dzs[z]/dzs[z-1]
                #
                dTdz = FiDiNonUniformDerivative1(
                    dFdz_T, dzs[z], DIFF1_T_SET, _zR_b)
                # d2Fdz2
                d2Tdz2 = FiDiNonUniformDerivative2(
                    d2Fdz2_T, dzs[z], DIFF2_T_SET_G, _zR_c)
            elif z == zNo - 1:
                # BC2
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # forward
                T_f = T_b
                T_ff = 0
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_BC2)
            else:
                # interior nodes
                # forward
                T_f = T_z[z+1]
                T_ff = T_z[z+2] if z < zNo-2 else 0
                # backward
                T_b = T_z[z-1]
                T_bb = T_z[z-2]
                # function value
                dFdz_T = [T_b, T_c, T_f]
                d2Fdz2_T = [T_bb, T_b, T_c, T_f, T_ff]
                # REVIEW
                ### uniform nodes ###
                # dFdz
                dTdz = FiDiDerivative1(dFdz_T, dz, DIFF1_T_SET)
                # d2Fdz2
                d2Tdz2 = FiDiDerivative2(d2Fdz2_T, dz, DIFF2_T_SET_G)

            # REVIEW
            # cal differentiate
            # backward difference
            # dTdz = (T_c - T_b)/(1*dz)
            # convective term
            _convectiveTerm = -1*v_z_DiLeVa*GaDe_DiLeVa*GaCpMeanMix_DiLeVa*dTdz
            # central difference
            # d2Tdz2 = (T_b - 2*T_c + T_f)/(dz**2)
            # dispersion flux [kJ/m^3.s]
            # _dispersionFluxT = (GaThCoEff*d2Tdz2)*1e-3
            _dispersionFluxT = ((1/PeNuHe0)*GaThCoEff_DiLeVa*d2Tdz2)*1
            # temperature in the catalyst surface [K]
            # Ts_cat
            # outward flux [kJ/m^2.s]
            _inwardFluxT = HeTrCo*SpSuAr*Tf*(Ts_r_cat_gas - T_c)*1e-3
            # total heat transfer between gas and solid [kJ/m^3.s]
            _heTrBeGaSoTerm = (1/GaHeCoTe0)*_inwardFluxT
            # heat exchange term [kJ/m^3.s] -> [no unit]
            _heatExchangeTerm = (1/GaHeCoTe0)*Qm
            # convective flux, diffusive flux, enthalpy of reaction, cooling heat

            # steady-state
            dxdt_T = _convectiveTerm + _dispersionFluxT + _heTrBeGaSoTerm + _heatExchangeTerm
            dxdtMat[indexT][0][z] = dxdt_T

            # dC/dt list
            # convert
            # solid thermal conductivity - [J/s.m.K] => [kJ/s.m.K]
            SoThCoEff_Conv = CaPo*SoThCoMix0/1000
            # overall heat of reaction - OvHeReT [kJ/m^3.s]
            OvHeReT_Conv = -1*OvHeReT
            # heat transfer coefficient - HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
            HeTrCo_Conv = HeTrCo/1000

            # loop vars
            _alpha = rf/SoThCoEff_Conv
            _beta = -1*HeTrCo_Conv/SoThCoEff_DiLeVa
            _DiLe = _alpha*_beta
            _H = (1-BeVoFr)*OvHeReT_Conv

            # set
            dxdtMat[indexT][1][z] = _H - _inwardFluxT

        # NOTE
        # flat
        dxdt = dxdtMat.flatten().tolist()

        return dxdt

# FIXME

    def modelReactions(P, T, y, CaBeDe):
        ''' 
        reaction rate expression list [kmol/m3.s]
        args: 
            P: pressure [Pa]
            T: temperature [K]
            y: mole fraction
            CaBeDe: catalyst bed density [kgcat/m^3 bed or particle]
        output: 
            r: reaction rate at T,P [kmol/m^3.s]
        '''
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
            r1 = K1*(ra1/(MATH.pow(ra2, 3)))*(1-ra3)*CaBeDe
            ra4 = PH2O - (1/KP2)*((PCO2*PH2)/PCO)
            r2 = K2*(1/ra2)*ra4*CaBeDe
            ra5 = (MATH.pow(PCH3OH, 2)/PH2O)-(PCH3OCH3/KP3)
            r3 = K3*ra5*CaBeDe

            # result
            # r = roundNum([r1, r2, r3], REACTION_RATE_ACCURACY)
            r = [r1, r2, r3]
            return r
        except Exception as e:
            print(e)
            raise
