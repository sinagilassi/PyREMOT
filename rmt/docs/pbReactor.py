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
from solvers.solSetting import solverSetting


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

    def runM1(self):
        """
        M1 modeling case
        steady-state modeling 
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

        # mole fraction
        MoFri = np.array(self.modelInput['feed']['mole-fraction'])
        # flowrate [mol/s]
        MoFlRa = self.modelInput['feed']['molar-flowrate']
        # component flowrate [mol/s]
        MoFlRai = MoFlRa*MoFri
        # flux [mol/m^2.s]
        MoFl = MoFlRa/CrSeAr
        # component flux [mol/m^2.s]
        MoFli = MoFl*MoFri

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # var no (Fi,FT,T,P)
        varNo = compNo + 3

        # initial values
        IV = np.zeros(varNo)
        IV[0:compNo] = MoFlRai
        IV[indexFlux] = MoFl
        IV[indexTemp] = T
        IV[indexPressure] = P
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
            "ExHe": ExHe

        }

        # save data
        timesNo = solverSetting['S2']['timesNo']

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
        # convert molar flowrate to mole fraction
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
                            "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

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
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # exchange heat spec ->
        ExHe = FunParam['ExHe']

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

        # superficial gas velocity [m/s]
        SuGaVe = rmtUtil.calSuperficialGasVelocityFromEOS(MoFl, P, T)

        # mixture molecular weight [kg/mol]
        MiMoWe = rmtUtil.mixtureMolecularWeight(MoFri, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe = calDensityIG(MiMoWe, CoSp)
        GaDeEOS = calDensityIGFromEOS(P, T, MiMoWe)

        # bed porosity (bed void fraction)
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']

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
        # forward frequency factor
        # A1 = 8.2e14
        # # forward activation energy [J/mol]
        # E1 = 284.5e3
        # # rate constant [1/s]
        # kFactor = 1e7
        # k1 = A1*np.exp(-E1/(R_CONST*T))*kFactor
        # # net reaction rate expression [mol/m^3.s]
        # r0 = k1*CoSpi[0]
        # Ri = [r0]

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
        const_F1 = BeVoFr/CrSeAr
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

    def runM2(self):
        """
        M2 modeling case
        dynamic model
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
                # if j == 0:
                #     IV2D[i][j] = SpCoi0[i] - (1/100)*SpCoi0[i]
                # else:
                #     IV2D[i][j] = SpCoi0[i] - (1/100)*SpCoi0[i]

        for j in range(zNo):
            IV2D[indexTemp][j] = T
            # if j == 0:
            #     IV2D[indexTemp][j] = T - 1
            # else:
            #     IV2D[indexTemp][j] = T - 1

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
        ssModelingResult = np.loadtxt('ssModeling.txt', dtype=np.float64)
        # ssdataXs = np.linspace(0, ReLe, zNo)
        ssXYList = pltc.plots2DSetXYList(dataXs, ssModelingResult)
        ssdataList = pltc.plots2DSetDataList(ssXYList, labelList)
        # datalists
        ssdataLists = [ssdataList[0:compNo],
                       ssdataList[indexTemp]]
        # subplot result
        pltc.plots2DSub(ssdataLists, "Reactor Length (m)",
                        "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

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
                                "Concentration (mol/m^3)", "1D Plug-Flow Reactor")

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
        pltc.plots2DSub(_dataListsSelected, "Reactor Length (m)",
                        "Concentration (mol/m^3)", "Dynamic Modeling of 1D Plug-Flow Reactor")

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
                dCdz = (Ci_c - Ci_b)/(dz)
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
            dTdz = (T_c - T_b)/(dz)

            dxdt_T = const_T2*(-const_T1*dTdz + (-OvHeReT + Qm))
            dxdtMat[indexT][z] = dxdt_T

        # flat
        dxdt = dxdtMat.flatten().tolist()

        return dxdt

    def modelReactions(P, T, y, CaBeDe):
        ''' 
        reaction rate expression list [kmol/m3.s]
        args: 
            P: pressure [Pa]
            T: temperature [K]
            y: mole fraction
            CaBeDe: catalyst bed density [kgcat/m^3 bed]
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
