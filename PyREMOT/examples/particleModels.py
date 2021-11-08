# DIFFUSION INTO CATALYST PARTICLE
# ---------------------------------

# import packages/modules
import math as MATH
import numpy as np
from numpy.lib import math
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from scipy import optimize
from scipy.optimize import fsolve
# internal
# data
from PyREMOT.data.inputDataReactor import *
# library
from PyREMOT.library.plot import plotClass as pltc
# core
from PyREMOT.core import constants as CONST
from PyREMOT.core.utilities import roundNum, selectFromListByIndex, selectRandomList
from PyREMOT.core.eqConstants import CONST_EQ_GAS_DIFFUSIVITY
# solvers
from PyREMOT.solvers.solSetting import solverSetting
from PyREMOT.solvers.solResultAnalysis import sortResult2
from PyREMOT.solvers.odeSolver import AdBash3, PreCorr3
from PyREMOT.solvers.solCatParticle import OrCoCatParticleClass
from PyREMOT.solvers.solCatParticle2 import FiElCatParticleClass
from PyREMOT.solvers.solOrCo import OrCoClass
from PyREMOT.solvers.solFiEl import FiElClass
from PyREMOT.solvers.solFiDi import FiDiBuildCMatrix, FiDiBuildTMatrix, FiDiBuildCMatrix_DiLe, FiDiBuildTMatrix_DiLe
# docs
from PyREMOT.docs.rmtReaction import reactionRateExe, componentFormationRate
from PyREMOT.docs.rmtUtility import rmtUtilityClass as rmtUtil
from PyREMOT.docs.rmtThermo import *
from PyREMOT.docs.modelSetting import MODEL_SETTING, PROCESS_SETTING
from PyREMOT.docs.gasTransPor import calTest
from PyREMOT.docs.gasTransPor import calGasDiffusivity
from PyREMOT.docs.gasTransPor import calGasViscosity, calMixturePropertyM1
from PyREMOT.docs.gasTransPor import calGasThermalConductivity


class ParticleModelClass:
    '''
    catalyst diffusion-reaction dynamic/steady-state models
    models:
        T1: dynamic 
        T2: steady-state
    '''

    def __init__(self, modelInput, internalData, reactionListSorted, reactionStochCoeffList):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted
        self.reactionStochCoeffList = reactionStochCoeffList

# NOTE
# dynamic model

    def runT1(self):
        """
        modeling case
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

        # numerical method
        numericalMethod = self.modelInput['test-const']['numerical-method']

        # NOTE
        # dimensionless length
        DiLeLe = 1

        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # length list
        dataXs = np.linspace(0, DiLeLe, zNo)
        # element size - dz [m]
        dz = DiLeLe/(zNo-1)
        if numericalMethod == "fdm":
            # finite difference points in the r direction
            rNo = solverSetting['ParticleModel']['rNo']['fdm']
        elif numericalMethod == "oc":
            # orthogonal collocation points in the r direction
            rNo = solverSetting['ParticleModel']['rNo']['oc']
        else:
            raise

        # length list
        dataRs = np.linspace(0, DiLeLe, rNo)

        # var no (Ci,T)
        varNo = compNo + \
            1 if processType != PROCESS_SETTING['ISO-THER'] else compNo
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of concentration layers
        noLayerC = compNo
        # number of temperature layers
        noLayerT = 1 if processType != PROCESS_SETTING['ISO-THER']else 0
        # number of layers
        noLayer = noLayerC + noLayerT
        # save all data
        noLayerSave = noLayerC + 1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = 1
        # number of var columns [i]
        varNoColumns = rNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(compNo):
            for i in range(rNo):
                # FIXME
                # solid phase
                # SpCoi0[m]/np.max(SpCoi0)
                IV2D[m][i] = 1e-6  # (0.1)*(SpCoi0[m]/np.max(SpCoi0))

        # check
        if processType != PROCESS_SETTING['ISO-THER']:
            # temperature [K]
            for i in range(varNoColumns):
                # solid phase
                IV2D[noLayer - 1][i] = 0

        # flatten IV
        IV = IV2D.flatten()

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

        # REVIEW
        # solver setting
        ReactionParams = {
            "reactionListSorted": reactionListSorted,
            "reactionStochCoeff": reactionStochCoeff
        }

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
                "noLayerC": noLayerC,
                "noLayerT": noLayerT,
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz,
            },
            "solverSetting": {
                "OrCoClassSetRes": OrCoClassSetRes
            },
            "reactionRateExpr": reactionRateExpr
        }

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

        # solid phase
        # mass convective term - (list) [kmol/m^3.s]
        _Cif = Cif if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.repeat(
            np.max(Cif), compNo)
        # mass diffusive term - (list)  [kmol/m^3.s]
        SoMaDiTe0 = (Dif*_Cif)/rf**2
        # heat diffusive term [kJ/m^3.s]
        SoHeDiTe0 = (GaThCoMix0*Tf/rf**2)*1e-3

        # mass transfer coefficient [m/s]
        MaTrCo0 = self.modelInput['test-const']['MaTrCo0']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo0 = self.modelInput['test-const']['HeTrCo0']

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "Dif": Dif,
            "Cpif": Cpif,
            "Cpf": Cpf,
            "rf": rf,
            "SoMaDiTe0": SoMaDiTe0,
            "SoHeDiTe0": SoHeDiTe0,
            "HeTrCo": HeTrCo0,
            "MaTrCo": MaTrCo0,
        }

        ### bulk properties ###
        # concentration in the bulk phase [kmol/m^3]
        Cbs = self.modelInput['test-const']['Cbi']
        # temperature in the bulk phase [K]
        Tb = self.modelInput['test-const']['Tb']

        # SoCpMeanMixEff [kJ/m^3.K]
        SoCpMeanMixEff = 279.34480838441203

        # -> concentration [-]
        Cbs_DiLeVa = np.zeros(compNo)
        for m in range(compNo):
            Cbs_DiLeVa[m] = Cbs[m]/np.max(Cbs)
        # -> temperature [-]
        Tb_DiLeVa = (Tb-Tf)/Tf

        # particle params
        ParticleParams = {
            "numericalMethod": numericalMethod,
            "SoCpMeanMixEff": SoCpMeanMixEff,
            "GaDii0": GaDii0,
            "Cbs": Cbs_DiLeVa,
            "Tb": Tb_DiLeVa
        }

        # time span
        tNo = solverSetting['ParticleModel']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['ParticleModel']['timesNo']

        # display result
        tNoDisplay = solverSetting['ParticleModel']['display']['tNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((noLayerSave, tNo, rNo))

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # FIXME
        n = solverSetting['T1']['ode-solver']['PreCorr3']['n']
        # t0 = 0
        # tn = 5
        # t = np.linspace(t0, tn, n+1)
        paramsSet = (ReactionParams, FunParam, ParticleParams,
                     DimensionlessAnalysisParams, processType)
        funSet = ParticleModelClass.modelEquationT1

        # time loop
        for i in range(tNo):
            # set time span
            t = np.array([opTSpan[i], opTSpan[i+1]])
            times = np.linspace(t[0], t[1], timesNo)

            # ode call
            # ode call
            if solverIVP == "AM":
                # adams moulton method
                sol = AdBash3(t[0], t[1], n, IV, funSet, paramsSet)
                # PreCorr3
                # sol = PreCorr3(t[0], t[1], n, IV, funSet, paramsSet)

                # ode result
                successStatus = True
                # time interval
                dataTime = t
                # all results
                # components, temperature layers
                dataYs = sol
            else:
                # method [1]: LSODA, [2]: BDF, [3]: Radau
                sol = solve_ivp(funSet, t, IV, method=solverIVP,
                                t_eval=times,  args=(paramsSet,))

                # ode result
                successStatus = sol.success
                # time interval
                dataTime = sol.t
                # all results
                # components layers
                dataYs = sol.y

            # check
            if successStatus is False:
                raise

            # last time result
            dataYs_End = dataYs[:, -1]

            # std format
            dataYs_Reshaped = np.reshape(
                dataYs_End, (noLayer, varNoColumns))
            # data
            # -> concentration
            dataYs_Concentration_DiLeVa = dataYs_Reshaped[:-
                                                          1] if processType != PROCESS_SETTING['ISO-THER'] else dataYs_Reshaped
            # -> temperature
            dataYs_Temperature_DiLeVa = dataYs_Reshaped[-1, :].reshape((1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
                0, rNo).reshape((1, varNoColumns))

            # sort out
            params1 = (compNo, noLayer, varNoRows, varNoColumns)
            params2 = (Cif, Tf, processType)
            dataYs_Sorted = sortResult2(
                dataYs_Reshaped, params1, params2)
            # component concentration [kmol/m^3]
            dataYs_Concentration_ReVa = dataYs_Sorted['data1']
            # temperature [K]
            dataYs_Temperature_ReVa = dataYs_Sorted['data2']

            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs_Concentration_ReVa, axis=0)
            dataYs1_MoFri = dataYs_Concentration_ReVa/dataYs1_Ctot

            # FIXME
            # build matrix
            dataYs_Combine = np.concatenate(
                (dataYs_Concentration_ReVa, dataYs_Temperature_ReVa), axis=0)

            dataYs_Combine_2 = np.concatenate(
                (dataYs_Concentration_DiLeVa, dataYs_Temperature_DiLeVa), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCo_DiLe": dataYs_Concentration_DiLeVa,
                "dataYCo": dataYs_Concentration_ReVa,
                "dataYMoFr": dataYs1_MoFri,
                "dataYT_DiLe": dataYs_Temperature_DiLeVa,
                "dataYT": dataYs_Temperature_ReVa,
                "dataY": dataYs_Combine
            })

            # save
            for m in range(compNo):
                # var list
                dataPacktime[m][i, :] = dataPack[i]['dataYCo_DiLe'][m, :]

            # temperature
            dataPacktime[indexTemp][i, :] = dataPack[i]['dataYT_DiLe'][:]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # plot info
        plotTitle = f"Dynamic Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # display result at specific time
        # subplot result
        xLabelSet = "Dimensionless Particle Radius"
        yLabelSet = "Dimensionless Concentration"

        for i in range(tNo):
            # var list
            _dataYs = dataPack[i]['dataY']
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataRs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo], dataList[indexTemp]]
            # dataLists = [dataList[0], dataList[1],
            #              dataList[2], dataList[indexTemp]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([0, 1], dataLists)
            if i == tNo-1:
                # subplot result
                pltc.plots2DSub(_dataListsSelected, xLabelSet,
                                yLabelSet, plotTitle)
                # pltc.plots2D(dataLists, "Dimensionless Particle Radius",
                #              "Concentration (mol/m^3)", plotTitle)

        # REVIEW
        # display result within time span
        _dataListsLoop = []
        _labelNameTime = []

        for i in range(varNo):
            # var list
            _dataPacktime = dataPacktime[i]
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataRs, _dataPacktime)
            # -> add label
            # build label
            for t in range(tNo):
                _opTSpanSet = float("{0:.2f}".format(opTSpan[t+1]))
                _name = labelList[i] + " at t=" + str(_opTSpanSet)
                # set
                _labelNameTime.append(_name)

            # random res at t
            # XYList_Random = np.random.choice(
            #     XYList[1:-1], tNoDisplay, replace=False)
            # XYList_Set = [XYList[0], *XYList_Random, XYList[-1]]

            randomizeRes = selectRandomList(XYList, tNoDisplay, _labelNameTime)
            # data random
            XYList_Set = randomizeRes['data1']
            _labelNameTime_Set = randomizeRes['data2']
            dataList = pltc.plots2DSetDataList(XYList_Set, _labelNameTime_Set)
            # datalists
            _dataListsLoop.append(dataList[0:tNo])
            # reset
            _labelNameTime = []

        # select time span

        # select datalist
        _dataListsSelected = selectFromListByIndex(
            [], _dataListsLoop)

        # subplot result
        pltc.plots2DSub(_dataListsSelected, "Dimensionless Particle Radius",
                        "Mole Fraction", "Dynamic Modeling of 1D Particle")

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationT1(t, y, paramsSet):
        '''
        T1 model
            mass, energy, and momentum balance equations
            modelParameters:
                ReactionParams:
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
                        noLayerC: number of layers for concentration
                        noLayerT: number of layers for temperature
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                    solverSetting:
                    reactionRateExpr: reaction rate expressions
                    DimensionlessAnalysisParams:
                    Cif: feed concentration [kmol/m^3]
                    Tf: feed temperature [K]
                    vf: feed superficial velocity [m/s]
                    zf: domain length [m]
                    Dif: diffusivity coefficient of component [m^2/s]
                    Cpif: feed heat capacity at constat pressure [kJ/kmol.K] | [J/mol.K]
                    GaMaCoTe0: feed mass convective term of gas phase [kmol/m^3.s]
                    GaMaDiTe0: feed mass diffusive term of gas phase [kmol/m^3.s]
                    GaHeCoTe0: feed heat convective term of gas phase [kJ/m^3.s]
                    GaHeDiTe0, feed heat diffusive term of gas phase [kJ/m^3.s]
                    ReNu0: Reynolds number
                    ScNu0: Schmidt number
                    ShNu0: Sherwood number
                    PrNu0: Prandtl number
                    PeNuMa0: mass Peclet number
                    PeNuHe0: heat Peclet number
                    MaTrCo: mass transfer coefficient - gas/solid [m/s]
                    HeTrCo: heat transfer coefficient - gas/solid [J/m^2.s.K]
                processType: isothermal/non-isothermal
        '''
        # parameters
        ReactionParams, FunParam, ParticleParams, DimensionlessAnalysisParams, processType = paramsSet
        # NOTE
        # reaction params
        reactionListSorted = ReactionParams['reactionListSorted']
        reactionStochCoeff = ReactionParams['reactionStochCoeff']

        # NOTE
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

        # NOTE
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

        # NOTE
        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # NOTE
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

        # NOTE
        # mesh setting
        meshSetting = FunParam['meshSetting']
        # number of layers for concentration
        noLayerC = meshSetting['noLayerC']
        # number of layers for temperature
        noLayerT = meshSetting['noLayerT']
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

        # NOTE
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

        # NOTE
        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # NOTE
        # particle parameters
        numericalMethod = ParticleParams['numericalMethod']
        SoCpMeanMixEff = ParticleParams['SoCpMeanMixEff']
        GaDii = ParticleParams['GaDii0']
        Cbs = ParticleParams['Cbs']
        Tb = ParticleParams['Tb']

        # NOTE
        # dimensionless analysis params
        #  feed concentration [kmol/m^3]
        Cif = DimensionlessAnalysisParams['Cif']
        # feed temperature
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # particle radius [m]
        rf = DimensionlessAnalysisParams['rf']
        # diffusivity coefficient of component [m^2/s]
        Dif = DimensionlessAnalysisParams['Dif']
        # feed heat capacity at constat pressure
        Cpif = DimensionlessAnalysisParams['Cpif']
        # feed mass diffusive term of solid phase [kmol/m^3.s]
        SoMaDiTe0 = DimensionlessAnalysisParams['SoMaDiTe0']
        # feed heat diffusive term of solid phase [kJ/m^3.s]
        SoHeDiTe0 = DimensionlessAnalysisParams['SoHeDiTe0']
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = DimensionlessAnalysisParams['MaTrCo']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = DimensionlessAnalysisParams['HeTrCo']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo

        # calculate
        # particle radius
        PaRa = PaDi/2

        # NOTE
        # pressure (constant)
        P_z = P0
        # temperature (constant)
        # Ts_r = T0*np.ones(rNo)

        # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
        SoThCoEff = CaThCo*((1 - CaPo)/CaTo)
        # dimensionless analysis
        SoCpMeanMixEff_ReVa = np.zeros(rNo)

        # NOTE
        ### yi manage ###
        fxMat = np.zeros((noLayer, varNoColumns))
        # reshape yj
        yj = np.array(y)
        yLoop = np.reshape(yj, (noLayer, varNoColumns))

        # concentration [kmol/m^3]
        SpCoi_mr = np.zeros((noLayerC, varNoColumns))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCoi_mr[m] = _SpCoi
        # shape
        CosSpiMatShape = (rNo, compNo)
        # concentration species in the solid phase [kmol/m^3]
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)

        # NOTE
        ### calculate ###
        # display concentration list in each oc point (rNo)
        for i in range(compNo):
            for r in range(rNo):
                _CosSpi_z = SpCoi_mr[i][r]
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

        # component mole fraction
        # mole fraction in the solid phase
        # MoFrsi_r0 = CosSpi_r/CosSp_r
        MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
            CosSpi_r_ReVa)

        # NOTE
        # temperature in solid phase - dimensionless
        Ts_r = np.zeros((varNoRows, varNoColumns))
        Ts_r = np.array(yLoop[noLayer-1]).reshape((varNoRows, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else \
            np.repeat(0, varNoColumns).reshape((varNoRows, varNoColumns))

        # temperature [K]
        Ts_r_ReVa0 = rmtUtil.calRealDiLessValue(Ts_r, Tf, "TEMP")
        Ts_r_ReVa = np.reshape(Ts_r_ReVa0, -1)

        # NOTE
        ## kinetics ##
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        SoCpMeanMix = np.zeros(rNo)
        # reaction term Ri [kmol/m^3.s]
        Ri_r = np.zeros((rNo, reactionListNo))
        # overall enthalpy
        OvHeReT = np.zeros(rNo)

        # REVIEW
        # ### physical & chemical properties ###
        # FIXME
        # solid gas thermal conductivity [J/s.m.K]
        SoThCoMix0 = GaThCoMix0
        # thermal conductivity - gas phase [J/s.m.K]
        # GaThCoi = np.zeros(compNo)  # f(T);
        GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest()
        # thermal conductivity - solid phase [J/s.m.K]
        # assume the same as gas phase
        # SoThCoi = np.zeros(compNo)  # f(T);
        SoThCoi = GaThCoi
        # mixture thermal conductivity - solid phase [J/s.m.K]
        SoThCoMix = GaThCoMix0
        # dimensionless analysis
        SoThCoMix_DiLeVa = SoThCoMix/SoThCoMix0
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
        SoDiiEff_DiLe = GaDii_DiLeVa

        # net reaction rate expression [kmol/m^3.s]
        # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
        for r in range(rNo):
            #
            _MoFrsi_r = MoFrsi_r[r, :]
            _CosSpi_r_ReVa = CosSpi_r_ReVa[r, :]
            _Ts_r_ReVa = Ts_r_ReVa[r]
            # loop
            loopVars0 = (_Ts_r_ReVa, P_z, _MoFrsi_r, _CosSpi_r_ReVa)

            # component formation rate [mol/m^3.s]
            # check unit
            r0 = np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))

            Ri_r[r, :] = r0

            # REVIEW
            # loop
            _Ri_r = Ri_r[r, :]
            # component formation rate [kmol/m^3.s]
            ri_r[r] = componentFormationRate(
                compNo, comList, reactionStochCoeff, _Ri_r)

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

        # ode eq [dy/dt] numbers
        for i in range(compNo):
            # central
            Ci_c = Cbs[i]
            # species concentration at different points of particle radius [rNo]
            # [Cs[3], Cs[2], Cs[1], Cs[0]]
            _Cs_r = CosSpi_r[:, i].flatten()
            # Cs[0], Cs[1], ...
            _Cs_r_Flip = np.flip(_Cs_r)
            # reaction term
            _ri_r = ri_r[:, i]
            _ri_r_Flip = np.flip(_ri_r)

            # dimensionless analysis
            if numericalMethod == "fdm":
                ### finite difference method ###
                # updated concentration gas-solid interface
                # loop var
                _dCsdtiVarLoop = (
                    SoDiiEff_DiLe[i], MaTrCo[i], _ri_r, Ci_c, CaPo, SoMaDiTe0[i], SoDiiEff[i], rf)

                # dC/dt list
                dCsdti = FiDiBuildCMatrix_DiLe(
                    compNo, PaRa, rNo, _Cs_r, _dCsdtiVarLoop, mode="test", fluxDir="lr")
            elif numericalMethod == "oc":
                ### orthogonal collocation method ###
                # updated concentration gas-solid interface
                # const
                _alpha = rf/GaDii0[i]
                _beta = MaTrCo[i]/GaDii_DiLeVa[i]
                _DiLeNu = _alpha*_beta
                _Ri = (1/SoMaDiTe0[i])*(1 - CaPo)*ri_r[:, i]
                # shape(rNo,1)
                _Cs_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                    _Cs_r, Ci_c, _DiLeNu)

                # dC/dt list
                dCsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                    _Cs_r_Updated, SoDiiEff_DiLe[i], _Ri, mode="test")
            else:
                pass

            # const
            _const1 = CaPo*(rf**2/GaDii0[i])
            _const2 = 1/_const1

            for r in range(rNo):
                # update
                fxMat[i][r] = _const2*dCsdti[r]

            # REVIEW
            ### solid phase ###
            # temperature at different points of particle radius [rNo]
            if processType != PROCESS_SETTING['ISO-THER']:

                # Ts[0], Ts[1], Ts[2], ...
                _Ts_r = Ts_r.flatten()
                # T[n], T[n-1], ..., T[0]
                # _Ts_r_Flip = np.flip(_Ts_r)

                # convert
                # [J/s.m.K] => [kJ/s.m.K]
                SoThCoEff_Conv = SoThCoMix0/1000
                # OvHeReT [kJ/m^3.s]
                OvHeReT_Conv = -1*OvHeReT
                # HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
                HeTrCo_Conv = HeTrCo/1000

                # dimensionless analysis
                if numericalMethod == "fdm":
                    ### finite difference method ###
                    # var loop
                    _dTsdtiVarLoop = (SoThCoEff_DiLeVa, HeTrCo_Conv,
                                      OvHeReT_Conv, Tb, CaPo, SoHeDiTe0, SoThCoEff_Conv, rf)

                    # dTs/dt list
                    dTsdti = FiDiBuildTMatrix_DiLe(
                        compNo, PaRa, rNo, _Ts_r, _dTsdtiVarLoop, mode="test")
                elif numericalMethod == "oc":
                    ### orthogonal collocation method ###
                    # loop vars
                    _alpha = rf/SoThCoEff_Conv
                    # FIXME
                    _beta = 1*HeTrCo_Conv/SoThCoEff_DiLeVa
                    _DiLeNu = _alpha*_beta
                    _H = (1/SoHeDiTe0)*(1 - CaPo)*OvHeReT_Conv
                    # T[n], T[n-1], ..., T[0]
                    # updated temperature gas--solid interface
                    _Ts_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                        _Ts_r, Tb, _DiLeNu)

                    # dTs/dt list
                    dTsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                        _Ts_r_Updated, SoThCoEff_DiLeVa, _H, mode="test")

                # const
                _const1 = SoCpMeanMixEff_ReVa*Tf/SoHeDiTe0
                _const2 = 1/_const1
                #
                for r in range(rNo):
                    # update
                    fxMat[indexT][r] = _const2[r]*dTsdti[r]

        # NOTE
        # flat
        dxdt = fxMat.flatten().tolist()

        # print
        print(f"time: {t} seconds")

        # return
        return dxdt

# NOTE
# steady-state

    def runT2(self):
        """
        steady state model
        unknowns: Cci, Tc (for catalyst)
            CT, GaDe = f(P, T, n)
        numerical method: 
            finite difference
            orthogonal collocation
            finite element
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

        # FIXME
        # concentration kmol/m^3.s
        ## inlet values ##
        # inlet volumetric flowrate at T,P [m^3/s]
        VoFlRa0 = self.modelInput['feed']['volumetric-flowrate']
        # inlet species concentration [mol/m^3]
        SpCoi0 = np.array(self.modelInput['feed']['concentration'])
        # inlet total concentration [mol/m^3]
        SpCo0 = np.sum(SpCoi0)

        # mole fraction in the gas phase
        MoFri0 = np.array(rmtUtil.moleFractionFromConcentrationSpecies(SpCoi0))

        # inlet superficial velocity [m/s]
        SuGaVe0 = VoFlRa0/CrSeAr
        # reaction rate expression
        reactionRateExpr = self.modelInput['reaction-rates']

        # component molecular weight [g/mol]
        MoWei = rmtUtil.extractCompData(self.internalData, "MW")
        # critical temperature [K]
        Tci = rmtUtil.extractCompData(self.internalData, "Tc")
        # critical pressure [Pa]
        Pci = rmtUtil.extractCompData(self.internalData, "Pc")

        # external heat
        ExHe = self.modelInput['external-heat']
        # cooling temperature [K]
        Tm = ExHe['MeTe']
        # overall heat transfer coefficient [J/s.m2.K]
        U = ExHe['OvHeTrCo']
        # heat transfer area over volume [m^2/m^3]
        a = 4/ReInDi

        # diffusivity coefficient - gas phase [m^2/s]
        GaDii0_paramsData = {
            "MoFri": MoFri0,
            "T": T,
            "P": P,
            "MWi": MoWei,
            "CrTei": Tci,
            "CrPri": Pci
        }
        # diffusivity coefficient of components in the mixture
        GaDii0 = calGasDiffusivity(
            CONST_EQ_GAS_DIFFUSIVITY['Chapman-Enskog'], compList, GaDii0_paramsData)

        # gas viscosity [Pa.s]
        GaVii0 = calGasViscosity(compList, T)
        # gas mixture viscosity [Pa.s]
        GaMiVi = calMixturePropertyM1(compNo, GaVii0, MoFri0, MoWei)

        # thermal conductivity - gas phase [J/s.m.K]
        GaThCoi0 = calGasThermalConductivity(compList, T)
        # mixture thermal conductivity - gas phase [J/s.m.K]
        GaThCoMix0 = calMixturePropertyM1(compNo, GaThCoi0, MoFri0, MoWei)

        ### calculation ###

        # mixture molecular weight [kg/mol]
        MiMoWe0 = rmtUtil.mixtureMolecularWeight(MoFri0, MoWei, "kg/mol")

        # gas density [kg/m^3]
        GaDe0 = calDensityIG(MiMoWe0, SpCo0)

        # heat capacity at constant pressure of mixture Cp [kJ/kmol.K] | [J/mol.K]
        # Cp mean list
        GaCpMeanList0 = calMeanHeatCapacityAtConstantPressure(compList, T)
        # Cp mixture
        GaCpMeanMix0 = calMixtureHeatCapacityAtConstantPressure(
            MoFri0, GaCpMeanList0)

        # numerical method
        numericalMethod = self.modelInput['test-const']['numerical-method']

        # REVIEW
        OrCoClassSetRes = []
        FiElClassInitRes = []

        # NOTE
        # dimensionless length
        DiLeLe = 1

        # finite difference points in the z direction
        zNo = solverSetting['S2']['zNo']
        # length list
        dataXs = np.linspace(0, DiLeLe, zNo)
        # element size - dz [m]
        dz = DiLeLe/(zNo-1)
        if numericalMethod == "fdm":
            # finite difference points in the r direction
            rNo = solverSetting['ParticleModel']['rNo']['fdm']
            # length list
            dataRs = np.linspace(0, DiLeLe, rNo)
        elif numericalMethod == "oc":
            # REVIEW
            # solver setting
            # orthogonal collocation method
            OrCoClassSet = OrCoClass()
            OrCoClassSetRes = OrCoClassSet.buildMatrix()
            # orthogonal collocation points in the r direction
            rNo = solverSetting['ParticleModel']['rNo']['oc']
            # length list
            dataRs = OrCoClassSet.Xc
        elif numericalMethod == "fem":
            # number of finite elements in the r direction
            NuEl = solverSetting['ParticleModel']['NuEl']
            # REVIEW
            # solver setting
            # finite element method
            # init finite element class
            FiElClassInit = FiElClass(NuEl)
            # res
            FiElClassInitRes = FiElClassInit.initFiEl()
            # orthogonal collocation points in the r direction
            rNo = FiElClassInitRes['NuToCoPo']
            # length list
            dataRs = FiElClassInitRes['xi']
        else:
            raise

        # var no (Ci,T)
        varNo = compNo + \
            1 if processType != PROCESS_SETTING['ISO-THER'] else compNo
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # concentration in solid phase
        varNoConInSolidBlock = rNo*compNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of concentration layers
        noLayerC = compNo
        # number of temperature layers
        noLayerT = 1 if processType != PROCESS_SETTING['ISO-THER']else 0
        # number of layers
        noLayer = noLayerC + noLayerT
        # save all data
        noLayerSave = noLayerC + 1
        # var no in each layer
        varNoLayer = zNo*(rNo+1)
        # total number of vars (Ci,T,Cci,Tci)
        varNoLayerT = noLayer*varNoLayer
        # concentration var number
        varNoCon = compNo*varNoLayer
        # number of var rows [j]
        varNoRows = 1
        # number of var columns [i]
        varNoColumns = rNo

        # initial values at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        BMatrixShape = (noLayer, varNoColumns)
        BUp2D = np.zeros(BMatrixShape)
        BLower2D = np.zeros(BMatrixShape)
        # initialize IV2D
        # -> concentration [mol/m^3] -> [-]
        for m in range(compNo):
            for i in range(rNo):
                # FIXME
                # solid phase
                IV2D[m][i] = SpCoi0[m]/np.max(SpCoi0)
                # set bounds
                BUp2D[m][i] = 1
                BLower2D[m][i] = 0

        # check
        if processType != PROCESS_SETTING['ISO-THER']:
            # temperature [K]
            for i in range(varNoColumns):
                # solid phase
                IV2D[noLayer - 1][i] = 0
                # set bounds
                BUp2D[noLayer - 1][i] = 1
                BLower2D[noLayer - 1][i] = -1

        # flatten IV
        IV = IV2D.flatten()
        BUp = BUp2D.flatten()
        BLower = BLower2D.flatten()

        # set bound
        setBounds = (BLower, BUp)

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
        ReactionParams = {
            "reactionListSorted": reactionListSorted,
            "reactionStochCoeff": reactionStochCoeff
        }

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
            "ExHe": {
                "OvHeTrCo": U,
                "EfHeTrAr": a,
                "MeTe": Tm
            },
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
                "noLayerC": noLayerC,
                "noLayerT": noLayerT,
                "noLayer": noLayer,
                "varNoLayer": varNoLayer,
                "varNoLayerT": varNoLayerT,
                "varNoRows": varNoRows,
                "varNoColumns": varNoColumns,
                "rNo": rNo,
                "zNo": zNo,
                "dz": dz,
            },
            "solverSetting": {
                "OrCoClassSetRes": OrCoClassSetRes,
                "FiElClassInitRes": FiElClassInitRes
            },
            "reactionRateExpr": reactionRateExpr
        }

        # NOTE
        ### dimensionless analysis ###
        # concentration [mol/m^3]
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

        # solid phase
        # mass convective term - (list) [mol/m^3.s]
        _Cif = Cif if MODEL_SETTING['GaMaCoTe0'] != "MAX" else np.repeat(
            np.max(Cif), compNo)
        # mass diffusive term - (list)  [mol/m^3.s]
        SoMaDiTe0 = (Dif*_Cif)/rf**2
        # heat diffusive term [kJ/m^3.s]
        SoHeDiTe0 = (GaThCoMix0*Tf/rf**2)*1e-3

        # mass transfer coefficient [m/s]
        MaTrCo0 = self.modelInput['test-const']['MaTrCo0']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo0 = self.modelInput['test-const']['HeTrCo0']

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "Dif": Dif,
            "Cpif": Cpif,
            "Cpf": Cpf,
            "rf": rf,
            "SoMaDiTe0": SoMaDiTe0,
            "SoHeDiTe0": SoHeDiTe0,
            "HeTrCo": HeTrCo0,
            "MaTrCo": MaTrCo0,
        }

        ### bulk properties ###
        # concentration in the bulk phase [kmol/m^3]
        Cbs = self.modelInput['test-const']['Cbi']
        # temperature in the bulk phase [K]
        Tb = self.modelInput['test-const']['Tb']

        # SoCpMeanMixEff [kJ/m^3.K]
        SoCpMeanMixEff = 279.34480838441203

        # -> concentration [-]
        Cbs_DiLeVa = np.zeros(compNo)
        for m in range(compNo):
            Cbs_DiLeVa[m] = Cbs[m]/np.max(Cbs)
        # -> temperature [-]
        Tb_DiLeVa = (Tb-Tf)/Tf

        # particle params
        ParticleParams = {
            "numericalMethod": numericalMethod,
            "SoCpMeanMixEff": SoCpMeanMixEff,
            "GaDii0": GaDii0,
            "Cbs": Cbs_DiLeVa,
            "Tb": Tb_DiLeVa
        }

        # time span
        tNo = solverSetting['ParticleModel']['tNo']
        opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        timesNo = solverSetting['ParticleModel']['timesNo']

        # display result
        tNoDisplay = solverSetting['ParticleModel']['display']['tNo']

        # result
        dataPack = []

        # build data list
        # over time
        dataPacktime = np.zeros((noLayerSave, tNo, rNo))

        # solver selection
        # BDF, Radau, LSODA
        solverIVP = "LSODA" if solverIVPSet == 'default' else solverIVPSet

        # FIXME
        n = solverSetting['T1']['ode-solver']['PreCorr3']['n']
        # t0 = 0
        # tn = 5
        # t = np.linspace(t0, tn, n+1)
        paramsSet = (ReactionParams, FunParam, ParticleParams,
                     DimensionlessAnalysisParams, processType)
        funSet = ParticleModelClass.modelEquationT2

        # NOTE
        ### solve a system of nonlinear algebraic equation ###
        if solverRootSet == "fsolve":
            sol = optimize.fsolve(funSet, IV, args=(paramsSet,))
            # result
            successStatus = True if len(sol) > 0 else False
            # all results
            # components, temperature layers
            dataYs = sol
        elif solverRootSet == "root":
            # root
            # lm, krylov, anderson, hybr, broyden1, linearmixing, diagbroyden, excitingmixing
            sol = optimize.root(funSet, IV, args=(
                paramsSet,), method='lm')
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x
        elif solverRootSet == "least_squares":
            sol = optimize.least_squares(
                funSet, IV, bounds=setBounds, args=(paramsSet,))
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x
        elif solverRootSet == "minimize":
            sol = optimize.minimize(
                funSet, IV, args=(paramsSet,), bounds=setBounds)
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x

        # check
        if successStatus is False:
            raise

        # last time result
        dataYs_End = dataYs

        # std format
        dataYs_Reshaped = np.reshape(
            dataYs_End, (noLayer, varNoColumns))
        # data
        # -> concentration
        dataYs_Concentration_DiLeVa = dataYs_Reshaped[:-
                                                      1] if processType != PROCESS_SETTING['ISO-THER'] else dataYs_Reshaped
        # -> temperature
        dataYs_Temperature_DiLeVa = dataYs_Reshaped[-1, :].reshape((1, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            0, rNo).reshape((1, varNoColumns))

        # sort out
        params1 = (compNo, noLayer, varNoRows, varNoColumns)
        params2 = (Cif, Tf, processType)
        dataYs_Sorted = sortResult2(
            dataYs_Reshaped, params1, params2)
        # component concentration [kmol/m^3]
        dataYs_Concentration_ReVa = dataYs_Sorted['data1']
        # temperature [K]
        dataYs_Temperature_ReVa = dataYs_Sorted['data2']

        # REVIEW
        # convert concentration to mole fraction
        dataYs1_Ctot = np.sum(dataYs_Concentration_ReVa, axis=0)
        dataYs1_MoFri = dataYs_Concentration_ReVa/dataYs1_Ctot

        # FIXME
        # build matrix
        dataYs_Combine_1 = np.concatenate(
            (dataYs_Concentration_ReVa, dataYs_Temperature_ReVa), axis=0)

        dataYs_Combine_2 = np.concatenate(
            (dataYs_Concentration_DiLeVa, dataYs_Temperature_DiLeVa), axis=0)

        # save data
        dataPack.append({
            "successStatus": successStatus,
            "dataTime": [],
            "dataYCo_DiLe": dataYs_Concentration_DiLeVa,
            "dataYCo": dataYs_Concentration_ReVa,
            "dataYMoFr": dataYs1_MoFri,
            "dataYT_DiLe": dataYs_Temperature_DiLeVa,
            "dataYT": dataYs_Temperature_ReVa,
            "dataY": dataYs_Combine_1
        })

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # plot info
        plotTitle = f"Steady-State Modeling for opT: {opT} with zNo: {zNo}, tNo: {tNo} within {elapsed} seconds"

        # REVIEW
        # post-processing result
        xLabelSet = "Dimensionless Particle Radius"
        yLabelSet = "Dimensionless Concentration"
        # var list
        _dataYs = dataPack[0]['dataY']
        # plot setting: build (x,y) series
        XYList = pltc.plots2DSetXYList(dataRs, _dataYs)
        # -> add label
        dataList = pltc.plots2DSetDataList(XYList, labelList)
        # datalists
        # dataLists = [dataList[0:compNo], dataList[indexTemp]]
        dataLists = [dataList[0], dataList[1],
                     dataList[2], dataList[3], dataList[4], dataList[5], dataList[indexTemp]]
        # select datalist
        _dataListsSelected = selectFromListByIndex([], dataLists)
        # subplot result
        pltc.plots2DSub(_dataListsSelected, xLabelSet,
                        yLabelSet, plotTitle)

        # return
        res = {
            "XYList": XYList,
            "dataList": dataList
        }

        return res

    def modelEquationT2(y, paramsSet):
        '''
        T1 model
            mass, energy, and momentum balance equations
            modelParameters:
                ReactionParams:
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
                        noLayerC: number of layers for concentration
                        noLayerT: number of layers for temperature
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [m]
                    solverSetting:
                        FiElClassInitRes
                    reactionRateExpr: reaction rate expressions
                    DimensionlessAnalysisParams:
                    Cif: feed concentration [kmol/m^3]
                    Tf: feed temperature [K]
                    vf: feed superficial velocity [m/s]
                    zf: domain length [m]
                    Dif: diffusivity coefficient of component [m^2/s]
                    Cpif: feed heat capacity at constat pressure [kJ/kmol.K] | [J/mol.K]
                    GaMaCoTe0: feed mass convective term of gas phase [kmol/m^3.s]
                    GaMaDiTe0: feed mass diffusive term of gas phase [kmol/m^3.s]
                    GaHeCoTe0: feed heat convective term of gas phase [kJ/m^3.s]
                    GaHeDiTe0, feed heat diffusive term of gas phase [kJ/m^3.s]
                    ReNu0: Reynolds number
                    ScNu0: Schmidt number
                    ShNu0: Sherwood number
                    PrNu0: Prandtl number
                    PeNuMa0: mass Peclet number
                    PeNuHe0: heat Peclet number
                    MaTrCo: mass transfer coefficient - gas/solid [m/s]
                    HeTrCo: heat transfer coefficient - gas/solid [J/m^2.s.K]
                processType: isothermal/non-isothermal
        '''
        # parameters
        ReactionParams, FunParam, ParticleParams, DimensionlessAnalysisParams, processType = paramsSet
        # NOTE
        # reaction params
        reactionListSorted = ReactionParams['reactionListSorted']
        reactionStochCoeff = ReactionParams['reactionStochCoeff']

        # NOTE
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

        # NOTE
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

        # NOTE
        # exchange heat spec ->
        ExHe = FunParam['ExHe']
        # var no. (concentration, temperature)
        varNo = const['varNo']
        # var no. in the domain
        varNoT = const['varNoT']

        # NOTE
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

        # NOTE
        # mesh setting
        meshSetting = FunParam['meshSetting']
        # number of layers for concentration
        noLayerC = meshSetting['noLayerC']
        # number of layers for temperature
        noLayerT = meshSetting['noLayerT']
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

        # NOTE
        # particle parameters
        numericalMethod = ParticleParams['numericalMethod']
        SoCpMeanMixEff = ParticleParams['SoCpMeanMixEff']
        GaDii = ParticleParams['GaDii0']
        Cbs = ParticleParams['Cbs']
        Tb = ParticleParams['Tb']

        # NOTE
        # solver setting
        solverSetting = FunParam['solverSetting']
        if numericalMethod == "oc":
            # number of collocation points
            ocN = solverSetting['OrCoClassSetRes']['N']
            ocXc = solverSetting['OrCoClassSetRes']['Xc']
            ocA = solverSetting['OrCoClassSetRes']['A']
            ocB = solverSetting['OrCoClassSetRes']['B']
            ocQ = solverSetting['OrCoClassSetRes']['Q']
            # init OrCoCatParticle
            OrCoCatParticleClassSet = OrCoCatParticleClass(
                ocXc, ocN, ocQ, ocA, ocB, varNo)
        elif numericalMethod == "fem":
            FiElClassInitRes = solverSetting['FiElClassInitRes']
            NuEl = FiElClassInitRes['NuEl']
            NuToCoPo = FiElClassInitRes['NuToCoPo']
            hi = FiElClassInitRes['hi']
            li = FiElClassInitRes['li']
            xi = FiElClassInitRes['xi']
            Xc = FiElClassInitRes['Xc']
            N = FiElClassInitRes['N']
            Q = FiElClassInitRes['Q']
            A = FiElClassInitRes['A']
            B = FiElClassInitRes['B']
            # init FiElCatParticleClass
            FiElCatParticleClassSet = FiElCatParticleClass(
                NuEl, NuToCoPo, hi, li, Xc, N, Q, A, B, varNo)

        # NOTE
        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # NOTE
        # dimensionless analysis params
        #  feed concentration [kmol/m^3]
        Cif = DimensionlessAnalysisParams['Cif']
        # feed temperature
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # particle radius [m]
        rf = DimensionlessAnalysisParams['rf']
        # diffusivity coefficient of component [m^2/s]
        Dif = DimensionlessAnalysisParams['Dif']
        # feed heat capacity at constat pressure
        Cpif = DimensionlessAnalysisParams['Cpif']
        # feed mass diffusive term of solid phase [kmol/m^3.s]
        SoMaDiTe0 = DimensionlessAnalysisParams['SoMaDiTe0']
        # feed heat diffusive term of solid phase [kJ/m^3.s]
        SoHeDiTe0 = DimensionlessAnalysisParams['SoHeDiTe0']
        # mass transfer coefficient - gas/solid [m/s]
        MaTrCo = DimensionlessAnalysisParams['MaTrCo']
        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = DimensionlessAnalysisParams['HeTrCo']

        # components no
        # y: component molar flowrate, total molar flux, temperature, pressure
        compNo = len(comList)
        indexT = compNo

        # calculate
        # particle radius
        PaRa = PaDi/2

        # NOTE
        # pressure (constant)
        P_z = P0
        # temperature (constant)
        # Ts_r = T0*np.ones(rNo)

        # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
        SoThCoEff = CaThCo*((1 - CaPo)/CaTo)
        # dimensionless analysis
        SoCpMeanMixEff_ReVa = np.zeros(rNo)

        # NOTE
        ### yi manage ###
        fxMat = np.zeros((noLayer, varNoColumns))
        # reshape yj
        yj = np.array(y)
        yLoop = np.reshape(yj, (noLayer, varNoColumns))

        # concentration [kmol/m^3]
        SpCoi_mr = np.zeros((noLayerC, varNoColumns))
        # layer
        for m in range(compNo):
            # -> concentration [mol/m^3]
            _SpCoi = yLoop[m]
            SpCoi_mr[m] = _SpCoi
        # shape
        CosSpiMatShape = (rNo, compNo)
        # concentration species in the solid phase [kmol/m^3]
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)

        # NOTE
        ### calculate ###
        # display concentration list in each oc point (rNo)
        for i in range(compNo):
            for r in range(rNo):
                _CosSpi_z = SpCoi_mr[i][r]
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

        # component mole fraction
        # mole fraction in the solid phase
        # MoFrsi_r0 = CosSpi_r/CosSp_r
        MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
            CosSpi_r_ReVa)

        # NOTE
        # temperature in solid phase - dimensionless
        Ts_r = np.zeros((varNoRows, varNoColumns))
        Ts_r = np.array(yLoop[noLayer-1]).reshape((varNoRows, varNoColumns)) if processType != PROCESS_SETTING['ISO-THER'] else \
            np.repeat(0, varNoColumns).reshape((varNoRows, varNoColumns))

        # temperature [K]
        Ts_r_ReVa0 = rmtUtil.calRealDiLessValue(Ts_r, Tf, "TEMP")
        Ts_r_ReVa = np.reshape(Ts_r_ReVa0, -1)

        # NOTE
        ## kinetics ##
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        SoCpMeanMix = np.zeros(rNo)
        # reaction term Ri [kmol/m^3.s]
        Ri_r = np.zeros((rNo, reactionListNo))
        # overall enthalpy
        OvHeReT = np.zeros(rNo)

        # REVIEW
        # ### physical & chemical properties ###
        # FIXME
        # solid gas thermal conductivity [J/s.m.K]
        SoThCoMix0 = GaThCoMix0
        # thermal conductivity - gas phase [J/s.m.K]
        # GaThCoi = np.zeros(compNo)  # f(T);
        GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest()
        # thermal conductivity - solid phase [J/s.m.K]
        # assume the same as gas phase
        # SoThCoi = np.zeros(compNo)  # f(T);
        SoThCoi = GaThCoi
        # mixture thermal conductivity - solid phase [J/s.m.K]
        SoThCoMix = GaThCoMix0
        # dimensionless analysis
        SoThCoMix_DiLeVa = SoThCoMix/SoThCoMix0
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
        SoDiiEff_DiLe = GaDii_DiLeVa

        # net reaction rate expression [kmol/m^3.s]
        # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
        for r in range(rNo):
            #
            _MoFrsi_r = MoFrsi_r[r, :]
            _CosSpi_r_ReVa = CosSpi_r_ReVa[r, :]
            _Ts_r_ReVa = Ts_r_ReVa[r]
            # loop
            loopVars0 = (_Ts_r_ReVa, P_z, _MoFrsi_r, _CosSpi_r_ReVa)

            # component formation rate [mol/m^3.s]
            # check unit
            r0 = np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))

            Ri_r[r, :] = r0

            # REVIEW
            # loop
            _Ri_r = Ri_r[r, :]
            # component formation rate [kmol/m^3.s]
            ri_r[r] = componentFormationRate(
                compNo, comList, reactionStochCoeff, _Ri_r)

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

        # ode eq [dy/dt] numbers
        for i in range(compNo):
            # central
            Ci_c = Cbs[i]
            # species concentration at different points of particle radius [rNo]
            # Cs[0], Cs[1], ...
            _Cs_r = CosSpi_r[:, i].flatten()
            # reaction term
            _ri_r = ri_r[:, i]
            # reaction type
            _Ri = (1/SoMaDiTe0[i])*(1 - CaPo)*ri_r[:, i]

            # dimensionless analysis
            if numericalMethod == "fdm":
                ### finite difference method ###
                # updated concentration gas-solid interface
                # loop var
                _dCsdtiVarLoop = (
                    SoDiiEff_DiLe[i], MaTrCo[i], _Ri, Ci_c, CaPo, SoMaDiTe0[i], SoDiiEff[i], rf)

                # dC/dt list
                dCsdti = FiDiBuildCMatrix_DiLe(
                    compNo, PaRa, rNo, _Cs_r, _dCsdtiVarLoop, mode="test", fluxDir="lr")
            elif numericalMethod == "oc":
                ### orthogonal collocation method ###
                # updated concentration gas-solid interface
                # const
                _alpha = rf/GaDii0[i]
                _beta = MaTrCo[i]/GaDii_DiLeVa[i]
                _DiLeNu = _alpha*_beta
                # REVIEW
                # _Cs_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                #     _Cs_r, Ci_c, _DiLeNu)
                _varLoop = (Ci_c, _DiLeNu)

                # dC/dt list
                dCsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                    _Cs_r, SoDiiEff_DiLe[i], _Ri, _varLoop, mode="test")
            elif numericalMethod == "fem":
                ### finite element method ###
                # const
                _alpha = rf/GaDii0[i]
                _beta = MaTrCo[i]/GaDii_DiLeVa[i]
                _DiLeNu = _alpha*_beta*hi[-1]
                # REVIEW
                _varLoop = (Ci_c, _DiLeNu)

                # dC/dt list
                dCsdti = FiElCatParticleClassSet.buildMatrix(
                    _Cs_r, SoDiiEff_DiLe[i], _Ri, _varLoop, mode="test")
            else:
                raise

            # const
            _const1 = CaPo*(rf**2/GaDii0[i])
            _const2 = 1/_const1

            for r in range(rNo):
                # update
                fxMat[i][r] = _const2*dCsdti[r]

            # REVIEW
            ### solid phase ###
            # temperature at different points of particle radius [rNo]
            if processType != PROCESS_SETTING['ISO-THER']:

                # Ts[0], Ts[1], Ts[2], ...
                _Ts_r = Ts_r.flatten()

                # convert
                # [J/s.m.K] => [kJ/s.m.K]
                SoThCoEff_Conv = SoThCoMix0/1000
                # OvHeReT [kJ/m^3.s]
                OvHeReT_Conv = -1*OvHeReT
                # HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
                HeTrCo_Conv = HeTrCo/1000
                # enthalpy of reaction term
                _dHRi = (1/SoHeDiTe0)*(1 - CaPo)*OvHeReT_Conv

                # dimensionless analysis
                if numericalMethod == "fdm":
                    ### finite difference method ###
                    # var loop
                    _dTsdtiVarLoop = (SoThCoEff_DiLeVa, HeTrCo_Conv,
                                      _dHRi, Tb, CaPo, SoHeDiTe0, SoThCoEff_Conv, rf)

                    # dTs/dt list
                    dTsdti = FiDiBuildTMatrix_DiLe(
                        compNo, PaRa, rNo, _Ts_r, _dTsdtiVarLoop, mode="test")
                elif numericalMethod == "oc":
                    ### orthogonal collocation method ###
                    # loop vars
                    _alpha = rf/SoThCoEff_Conv
                    # FIXME
                    _beta = 1*HeTrCo_Conv/SoThCoEff_DiLeVa
                    _DiLeNu = _alpha*_beta
                    # _H = (1/SoHeDiTe0)*(1 - CaPo)*OvHeReT_Conv
                    # T[n], T[n-1], ..., T[0]
                    # updated temperature gas--solid interface
                    # _Ts_r_Updated = OrCoCatParticleClassSet.CalUpdateYnSolidGasInterface(
                    #     _Ts_r, Tb, _DiLeNu)
                    _varLoop = (Tb, _DiLeNu)

                    # dTs/dt list
                    dTsdti = OrCoCatParticleClassSet.buildOrCoMatrix(
                        _Ts_r, SoThCoEff_DiLeVa, _dHRi, _varLoop, mode="test")
                elif numericalMethod == "fem":
                    ### finite element method ###
                    # const
                    # loop vars
                    _alpha = rf/SoThCoEff_Conv
                    # FIXME
                    # ! outard flux
                    _beta = 1*HeTrCo_Conv/SoThCoEff_DiLeVa
                    _DiLeNu = _alpha*_beta*hi[-1]
                    # _H = (1/SoHeDiTe0)*(1 - CaPo)*OvHeReT_Conv
                    # T[n], T[n-1], ..., T[0]
                    _varLoop = (Tb, _DiLeNu)

                    # dTs/dt list
                    dTsdti = FiElCatParticleClassSet.buildMatrix(
                        _Ts_r, SoThCoEff_DiLeVa, _dHRi, _varLoop, mode="test")
                else:
                    raise

                # const
                _const1 = SoCpMeanMixEff_ReVa*Tf/SoHeDiTe0
                _const2 = 1/_const1
                #
                for r in range(rNo):
                    # update
                    fxMat[indexT][r] = _const2[r]*dTsdti[r]

        # NOTE
        # flat
        dxdt = fxMat.flatten().tolist()

        # return
        return dxdt
