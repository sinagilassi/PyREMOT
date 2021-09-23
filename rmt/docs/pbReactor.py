# PACKED-BED REACTOR MODEL
# -------------------------

# import packages/modules
import math as MATH
import numpy as np
from numpy.lib import math
from library.plot import plotClass as pltc
from scipy.integrate import solve_ivp
# internal
from core.errors import errGeneralClass as errGeneral
from data.inputDataReactor import *
from core import constants as CONST
from core.utilities import roundNum, selectFromListByIndex
from core.config import REACTION_RATE_ACCURACY
from .rmtUtility import rmtUtilityClass as rmtUtil
from .rmtThermo import *
from .rmtReaction import reactionRateExe
from solvers.solSetting import solverSetting
from .fluidFilm import *
from core.eqConstants import CONST_EQ_Sh
from solvers.solOrCo import OrCoClass
from solvers.solCatParticle import OrCoCatParticleClass


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
            "ExHe": ExHe,
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

        # ode call
        sol = solve_ivp(PackedBedReactorClass.modelEquationM1,
                        t, IV, method="LSODA", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

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

        # plot info
        plotTitle = f"Steady-State Modeling with timesNo: {timesNo}"

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
        Ri = 1000*np.array(PackedBedReactorClass.modelReactions(
            P, T, MoFri, CaBeDe))

        # using equation
        params0 = reactionRateExpr['PARAMS']
        varis0 = reactionRateExpr['VARS']
        rates0 = reactionRateExpr['RATES']
        # loop
        loopVars0 = (T, P, MoFri, CaBeDe)

        rDict = {
            "PARAMS": params0,
            "VARS": varis0,
            "RATES": rates0,
        }

        # Ri_expr = reactionRateExe(loopVars0, params0, varis0, rates0
        #                           )

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
        #

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            sol = solve_ivp(PackedBedReactorClass.modelEquationM2,
                            t, IV, method="BDF", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

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
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo}"

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
            ## kinetics ##
            # net reaction rate expression [kmol/m^3.s]
            # rf[kmol/kgcat.s]*CaBeDe[kgcat/m^3]
            r0 = np.array(PackedBedReactorClass.modelReactions(
                P_z[z], T_z[z], MoFri, CaBeDe))
            # loop
            Ri_z[z, :] = r0

            # FIXME
            #  H2
            # R_H2 = -(3*r0[0]-r0[1])
            # ri0[0] = R_H2
            # # CO2
            # R_CO2 = -(r0[0]-r0[1])
            # ri0[1] = R_CO2
            # # H2O
            # R_H2O = (r0[0]-r0[1]+r0[2])
            # ri0[2] = R_H2O
            # # CO
            # R_CO = -(r0[1])
            # ri0[3] = R_CO
            # # CH3OH
            # R_CH3OH = -(2*r0[2]-r0[0])
            # ri0[4] = R_CH3OH
            # # DME
            # R_DME = (r0[2])
            # ri0[5] = R_DME
            # # total
            # R_T = -(2*r0[0])

            # REVIEW
            # component formation rate [kmol/m^3.s]
            # ri = np.zeros(compNo)
            for k in range(compNo):
                # reset
                _riLoop = 0
                # number of reactions
                for m in range(len(reactionStochCoeff)):
                    # number of components in each reaction
                    for n in range(len(reactionStochCoeff[m])):
                        # check component id
                        if comList[k] == reactionStochCoeff[m][n][0]:
                            _riLoop += reactionStochCoeff[m][n][1]*Ri_z[z][m]
                    ri[k] = _riLoop

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
            if Tm == 0:
                # adiabatic
                Qm = 0
            else:
                # heat added/removed from the reactor
                # Tm > T: heat is added (positive sign)
                # T > Tm: heat removed (negative sign)
                Qm = (Ua*(Tm - T))*1e-3

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
        sol = solve_ivp(PackedBedReactorClass.modelEquationM3,
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

        # plot info
        plotTitle = f"Steady-State Modeling [M3] with timesNo: {timesNo}"

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
        # steady-state result
        # txt
        # ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # binary
        ssModelingResult = np.load('ResM1.npy')
        # ssdataXs = np.linspace(0, ReLe, zNo)
        ssXYList = pltc.plots2DSetXYList(dataX, ssModelingResult)
        ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        ssdataLists = [ssdataList[0:compNo],
                       ssdataList[indexTemp]]

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
                            "Concentration (mol/m^3)", plotTitle, ssdataLists)

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
        #

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            sol = solve_ivp(PackedBedReactorClass.modelEquationM5,
                            t, IV, method="BDF", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

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
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo}"

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

    def modelEquationM5(t, y, reactionListSorted, reactionStochCoeff, FunParam):
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
            r0 = np.array(PackedBedReactorClass.modelReactions(
                P_z[z], T_z[z], MoFri, CaBeDe))
            # loop
            Ri_z[z, :] = r0

            # FIXME
            #  H2
            # R_H2 = -(3*r0[0]-r0[1])
            # ri0[0] = R_H2
            # # CO2
            # R_CO2 = -(r0[0]-r0[1])
            # ri0[1] = R_CO2
            # # H2O
            # R_H2O = (r0[0]-r0[1]+r0[2])
            # ri0[2] = R_H2O
            # # CO
            # R_CO = -(r0[1])
            # ri0[3] = R_CO
            # # CH3OH
            # R_CH3OH = -(2*r0[2]-r0[0])
            # ri0[4] = R_CH3OH
            # # DME
            # R_DME = (r0[2])
            # ri0[5] = R_DME
            # # total
            # R_T = -(2*r0[0])

            # REVIEW
            # component formation rate [kmol/m^3.s]
            # ri = np.zeros(compNo)
            for k in range(compNo):
                # reset
                _riLoop = 0
                # number of reactions
                for m in range(len(reactionStochCoeff)):
                    # number of components in each reaction
                    for n in range(len(reactionStochCoeff[m])):
                        # check component id
                        if comList[k] == reactionStochCoeff[m][n][0]:
                            _riLoop += reactionStochCoeff[m][n][1]*Ri_z[z][m]
                    ri[k] = _riLoop

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
            if Tm == 0:
                # adiabatic
                Qm = 0
            else:
                # heat added/removed from the reactor
                # Tm > T: heat is added (positive sign)
                # T > Tm: heat removed (negative sign)
                Qm = (Ua*(Tm - T))*1e-3

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

        return dxdt

# NOTE
# dynamic heterogenous modeling

    def runM6(self):
        """
        M6 modeling case
        dynamic model
        unknowns: Ci, T (dynamic), P, v (static), Cci, Tc (dynamic, for catalyst)
            CT, GaDe = f(P, T, n)
        """

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
        #

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            # method [1]: LSODA, [2]: BDF
            sol = solve_ivp(PackedBedReactorClass.modelEquationM6,
                            t, IV, method="LSODA", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

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
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo}"

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
            GaViMix = 1e-5  # f(yi,GaVi,MWs);
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
            SoThCoEff = CaPo*SoThCoMix + (1 - CaPo)*CaThCo

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
                r0 = np.array(PackedBedReactorClass.modelReactions(
                    P_z[z], Ts_r[r], MoFrsi_r[r], CaDe))
                # loop
                Ri_zr[z, r, :] = r0
                Ri_r[r, :] = r0

                # reset
                _riLoop = 0

                # REVIEW
                # component formation rate [kmol/m^3.s]
                # ri = np.zeros(compNo)
                for k in range(compNo):
                    # reset
                    _riLoop = 0
                    # number of reactions
                    for m in range(len(reactionStochCoeff)):
                        # number of components in each reaction
                        for n in range(len(reactionStochCoeff[m])):
                            # check component id
                            if comList[k] == reactionStochCoeff[m][n][0]:
                                _riLoop += reactionStochCoeff[m][n][1] * \
                                    Ri_r[r][m]
                        ri_r[r][k] = _riLoop

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
            # effective heat capacity - gas phase [kJ/m^3.K]
            GaCpMeanMixEff = CoSp*GaCpMeanMix*BeVoFr

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
                GaCpMeanMixEff, GaViMix, GaThCoMix, MiMoWe)
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
            if Tm == 0:
                # adiabatic
                Qm = 0
            else:
                # heat added/removed from the reactor
                # Tm > T: heat is added (positive sign)
                # T > Tm: heat removed (negative sign)
                Qm = (Ua*(Tm - T))*1e-3

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
            const_T1 = MoFl*GaCpMeanMix
            const_T2 = 1/GaCpMeanMixEff

            # catalyst
            const_Cs1 = 1/(CaPo*(PaRa**2))
            const_Ts1 = 1/SoCpMeanMixEff

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
            # dispersion flux [kJ/m^3.s]
            _dispersionFluxT = (GaThCoEff*d2Tdz2)*1e-3
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
