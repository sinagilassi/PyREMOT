# PACKED-BED REACTOR MODEL
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
from test_rmt import ReLe
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

        # time span
        # t = (0.0, rea_L)
        t = np.array([0, ReLe])
        t_span = np.array([0, ReLe])
        times = np.linspace(t_span[0], t_span[1], 100)
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
        # flux
        dataYs2 = sol.y[indexFlux, :]
        labelListYs2 = labelList[indexFlux]
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
            dataLists = [dataList[0:compNo],
                         dataList[indexFlux], dataList[indexTemp], dataList[indexPressure]]
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

        # kinetics
        # Ri = np.array(PlugFlowReactorClass.modelReactions(P, T, MoFri))
        # forward frequency factor
        A1 = 8.2e14
        # forward activation energy [J/mol]
        E1 = 284.5e3
        # rate constant [1/s]
        kFactor = 1e7
        k1 = A1*np.exp(-E1/(R_CONST*T))*kFactor
        # net reaction rate expression [mol/m^3.s]
        r0 = k1*CoSpi[0]
        Ri = [r0]

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
                ri[k] = _riLoop*CaBeDe

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
        labelList.append("Total Concentration")
        labelList.append("Temperature")
        labelList.append("Pressure")

        # component no
        compNo = len(compList)
        indexTConcentration = compNo
        indexTemp = indexTConcentration + 1
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
        # inlet species concentration [mol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [mol/m^3]
        SpCo0 = np.sum(SpCoi0)

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")

        # external heat
        ExHe = self.modelInput['external-heat']

        # gas mixture viscosity [Pa.s]
        GaMiVi = self.modelInput['feed']['mixture-viscosity']

        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # dz
        dz = ReLe/zNo

        # var no (Ci,CT,T)
        varNo = compNo + 2
        # total var no along the reactor length
        varNoT = varNo*zNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (varNo, zNo)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration
        for i in range(compNo):
            for j in range(zNo):
                IV2D[i][j] = SpCoi0[i]

        # total concentration
        for j in range(zNo):
            IV2D[indexTConcentration][j] = SpCo0[i]

        for j in range(zNo):
            IV2D[indexTemp][j] = T

        # flatten IV
        IV = IV2D.flatten()

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
                "SpCo0": SpCo0
            }

        }

        # time span
        opTSpan = np.linspace(0, opT, zNo)

        # result
        dataPack = []

        # time loop
        for i in len(opTSpan) - 1:
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], 10)

            # ode call
            sol = solve_ivp(PackedBedReactorClass.modelEquationM2,
                            t, IV, method="LSODA", t_eval=times, args=(reactionListSorted, reactionStochCoeff, FunParam))

            # ode result
            successStatus = sol.success
            dataX = sol.t
            # all results
            dataYs = sol.y
            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataX": dataX,
                "dataYs": dataYs
            })

            # component concentration [mol/m^3]
            dataYs1 = sol.y[0:compNo, :]
            # total concentration
            dataYs2 = sol.y[indexTConcentration, :]
            # temperature
            dataYs3 = sol.y[indexTemp, :]

            # update initial values [IV]
            # at t > 0
            IV[0:compNo, :] = dataYs1
            IV[indexTConcentration, :] = dataYs2
            IV[indexTemp, :] = dataYs3

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataX, dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTConcentration], dataList[indexTemp], dataList[indexPressure]]
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
                        dz: 
                    ReSpec: reactor spec
                    ExHe: exchange heat spec
                        OvHeTrCo: overall heat transfer coefficient [J/m^2.s.K]
                        EfHeTrAr: effective heat transfer area [m^2]
                        MeTe: medium temperature [K]
                    constBC1: 
                        VoFlRa0: inlet volumetric flowrate [m^3/s],
                        SpCoi0: species concentration [mol/m^3],
                        SpCo0: total concentration [mol/m^3]

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
        # dz
        dz = const['dz']
        # reactor spec ->
        ReSpec = FunParam['ReSpec']
        # particle diameter [m]
        PaDi = ReSpec['PaDi']
        # bed void fraction - porosity
        BeVoFr = ReSpec['BeVoFr']
        # bulk density (catalyst bed density)
        CaBeDe = ReSpec['CaBeDe']

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
        # inlet species concentration [mol/m^3]
        SpCoi0 = constBC1['SpCoi0']
        # inlet total concentration [mol/m^3]
        SpCo0 = constBC1['SpCo0']

        # calculate
        # molar flowrate [mol/s]
        MoFlRa0 = SpCo0*VoFlRa0
        # superficial gas velocity [m/s]
        SuGaVe0 = VoFlRa0/CrSeAr

        # superficial gas velocity [m/s]
        SpCoList_z = np.zeros(zNo)
        SpCoList_z[0] = SuGaVe0

        # total molar flux [mol/m^2.s]
        MoFl_z = np.zeros(zNo)
        MoFl_z[0] = MoFlRa0

        # reaction rate
        Ri_z = np.zeros((zNo, reactionListNo))

        # FIXME
        # pressure [Pa]
        P_z = np.zeros(zNo)
        P_z[0] = 1

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexTConcentration = compNo
        indexT = indexTConcentration + 1
        indexP = indexT + 1

        # NOTE
        # distribute y[i] value through the reactor length
        # reshape
        yLoop = np.reshape(y, (varNo, zNo))

        # -> concentration [mol/m^3]
        SpCoi_z = np.array((compNo, zNo))
        for i in range(compNo):
            SpCoi_z[i, :] = yLoop[i, :]

        # total concentration
        CT_z = np.array((1, zNo))
        CT_z[1, :] = yLoop[indexTConcentration, :]

        # temperature [K]
        T_z = np.array((1, zNo))
        T_z[1, :] = yLoop[indexT, :]

        # species concentration [mol/m^3]
        CoSpi = []

        # NOTE
        # define ode equations for each finite difference [zNo]
        for z in range(zNo):
            ## block ##

            # concentration species [mol/m^3]
            for i in range(compNo):
                CoSpi.append(SpCoi_z[i][z])

            # total concentration [mol/m^3]
            CoSp = CT_z[z]
            # temperature [K]
            T = T_z[z]
            # pressure [Pa]
            P = P_z[z]

            ## calculate ##
            # mole fraction
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi))

            # total flowrate [mol/s]
            # [mol/m^3]*[m/s]*[m^2]
            MoFlRa = MoFl_z[m]
            # molar flowrate list [mol/m^3]
            MoFlRai = MoFlRa*MoFri

            # molar flux [mol/m^2.s]
            MoFl = MoFlRa/(CrSeAr*BeVoFr)

            # volumetric flowrate [m^3/s]
            VoFlRai = calVolumetricFlowrateIG(P, T, MoFlRai)

            # superficial gas velocity [m/s]
            SuGaVe = rmtUtil.calSuperficialGasVelocityFromEOS(MoFl, P, T)

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
            dxdt_P = -1*RHS_ergun
            # dxdt.append(dxdt_P)
            P_z[z+1] = dxdt_P*dz + P_z[z]

            # NOTE
            # kinetics
            # Ri = np.array(PlugFlowReactorClass.modelReactions(P, T, MoFri))
            # forward frequency factor
            A1 = 8.2e14
            # forward activation energy [J/mol]
            E1 = 284.5e3
            # rate constant [1/s]
            kFactor = 1e7
            k1 = A1*np.exp(-E1/(R_CONST*T))*kFactor
            # net reaction rate expression [mol/m^3.s]
            r0 = k1*CoSpi[0]
            Ri_z[z] = [r0]

            # component formation rate [mol/m^3.s]
            # rf[mol/kgcat.s]*CaBeDe[kgcat/m^3]
            ri = np.zeros(compNo)
            for k in range(compNo):
                # reset
                _riLoop = 0
                for m in range(len(reactionStochCoeff)):
                    for n in range(len(reactionStochCoeff[m])):
                        if comList[k] == reactionStochCoeff[m][n][0]:
                            _riLoop += reactionStochCoeff[m][n][1]*Ri_z[z][m]
                    ri[k] = _riLoop*CaBeDe

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
            EnChList = np.array(
                calEnthalpyChangeOfReaction(reactionListSorted, T))
            # heat of reaction at T [kJ/kmol] | [J/mol]
            HeReT = np.array(EnChList + StHeRe25)
            # overall heat of reaction [J/m^3.s]
            OvHeReT = np.dot(Ri_z[z], HeReT)

            # NOTE
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
            # dxdt = []
            # matrix
            dxdtMat = np.zeros((varNo, zNo))

            # loop vars
            const_F1 = 1/BeVoFr
            const_T1 = MoFl*CpMeanMixture
            const_T2 = 1/(CoSp*CpMeanMixture*BeVoFr)

            # concentration [mol/m^3]
            for i in range(compNo):
                # mass balance (forward difference)
                Ci_c = SpCoi_z[i][z]
                Ci_f = SpCoi_z[i][z + 1]
                #
                dxdt_F = const_F1*(-SuGaVe*((Ci_f - Ci_c)/dz) + ri[i])
                dxdtMat[i][z] = dxdt_F

            # total concentration [mol/m^3]
            CT_c = SpCoi_z[indexTConcentration][z]
            CT_f = SpCoi_z[indexTConcentration][z + 1]
            #
            dxdt_Fl = -SuGaVe*((CT_f - CT_c)/dz) + OvR
            dxdtMat[indexTConcentration][z] = dxdt_Fl

            # energy balance (temperature) [K]
            T_c = SpCoi_z[indexT][z]
            T_f = SpCoi_z[indexT][z + 1]
            #
            dxdt_T = const_T2*(-const_T1*((T_f - T_c)/dz) + (-OvHeReT + Qm))
            dxdtMat[indexT][z] = dxdt_T

            # flat
            dxdt = dxdtMat.flatten().tolist()

        return dxdt
