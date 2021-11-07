# GAS PHASE HEAT AND MASS BALANCES
# ---------------------------------

# import packages/modules
import math as MATH
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.lib import math
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from scipy import optimize

# internal
# core
from PyREMOT.core.utilities import roundNum, selectFromListByIndex
from PyREMOT.core import constants as CONST
# library
from PyREMOT.library.plot import plotClass as pltc
# docs
from PyREMOT.docs.modelSetting import MODEL_SETTING, PROCESS_SETTING
from PyREMOT.docs.rmtUtility import rmtUtilityClass as rmtUtil
from PyREMOT.docs.rmtThermo import *
from PyREMOT.docs.gasTransPor import calTest
from PyREMOT.docs.fluidFilm import *
from PyREMOT.docs.rmtReaction import reactionRateExe, componentFormationRate
# data
from PyREMOT.data.inputDataReactor import *
# solvers
from PyREMOT.solvers.solSetting import solverSetting
from PyREMOT.solvers.solResultAnalysis import setOptimizeRootMethod
from PyREMOT.solvers.solFiDi import FiDiMeshGenerator
from PyREMOT.solvers.solFiDi import FiDiBuildCMatrix, FiDiBuildTMatrix, FiDiDerivative1, FiDiDerivative2, FiDiNonUniformDerivative1, FiDiNonUniformDerivative2


class HomoModelClass:
    '''
    homogenous models
    catalyst diffusion-reaction dynamic/steady-state models
    '''

    def __init__(self, modelInput, internalData, reactionListSorted, reactionStochCoeffList):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted
        self.reactionStochCoeffList = reactionStochCoeffList

# NOTE

    def runT1(self):
        """
        modeling case: static model
        unknowns: Ci, T (dynamic), P, v (static), CT, GaDe = f(P, T, n)
        numerical method: finite difference
        """
        # start computation
        start = timer()

        # solver setting
        solverConfig = self.modelInput['solver-config']
        solverRootSet = solverConfig['root']
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

        # domain length
        DoLe = 1

        # orthogonal collocation points in the r direction
        rNo = solverSetting['T1']['rNo']
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
        # interstitial gas velocity [m/s]
        InGaVe0 = SuGaVe0/BeVoFr

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
        varNo = compNo + \
            1 if processType != PROCESS_SETTING['ISO-THER'] else compNo
        # concentration var no
        varNoCon = compNo*zNo
        # temperature var no
        varNoTemp = 1*zNo
        # total var no along the reactor length (in gas phase)
        varNoT = varNo*zNo

        # number of layers
        # concentration layer for each component C[m,j,i]
        # m: layer, j: row (rNo), i: column (zNo)

        # number of concentration layers
        noLayerC = compNo
        # number of temperature layers
        noLayerT = 1 if processType != PROCESS_SETTING['ISO-THER'] else 0
        # number of layers
        noLayer = noLayerC + noLayerT
        # var no in each layer
        varNoLayer = zNo
        # total number of vars (Ci,T)
        varNoLayerT = noLayer*varNoLayer
        # number of var columns [i]
        varNoColumns = zNo
        # number of var rows [j]
        varNoRows = 1

        # NOTE
        # initial guess at t = 0 and z >> 0
        IVMatrixShape = (noLayer, varNoRows, varNoColumns)
        IV2D = np.zeros(IVMatrixShape)
        # bounds
        BMatrixShape = (noLayer, varNoRows, varNoColumns)
        BUp2D = np.zeros(BMatrixShape)
        BLower2D = np.zeros(BMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayerC):
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # FIXME
                        # gas phase
                        IV2D[m][j][i] = 0.5  # SpCoi0[m]/np.max(SpCoi0)
                        # set bounds
                        BUp2D[m][j][i] = 1
                        BLower2D[m][j][i] = 0

        # check
        if processType != "iso-thermal":
            # temperature [K]
            for i in range(varNoColumns):
                for j in range(varNoRows):
                    # separate phase
                    if j == 0:
                        # FIXME
                        # gas phase
                        IV2D[noLayer - 1][j][i] = 0.25
                        # set bounds
                        BUp2D[noLayer - 1][j][i] = 1
                        BLower2D[noLayer - 1][j][i] = -1

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
        _PeNuHe0 = ReNu0*PrNu0
        _PeNuHe1 = GaHeCoTe0/GaHeDiTe0
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
                "GaThCoMix0": GaThCoMix0,
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
            "reactionRateExpr": reactionRateExpr,
        }

        # dimensionless analysis parameters
        DimensionlessAnalysisParams = {
            "Cif": Cif,
            "Tf": Tf,
            "vf": vf,
            "zf": zf,
            "Dif": Dif,
            "Cpif": Cpif,
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
            "HeTrCo": HeTrCo
        }

        # time span
        # tNo = solverSetting['S2']['tNo']
        # opTSpan = np.linspace(0, opT, tNo + 1)

        # save data
        # timesNo = solverSetting['S2']['timesNo']

        # result
        dataPack = []
        successStatus = False

        # build data list
        # over time
        # dataPacktime = np.zeros((varNo, tNo, zNo))

        # NOTE
        ### solve a system of nonlinear algebraic equation ###
        if solverRootSet == "fsolve":
            sol = optimize.fsolve(HomoModelClass.modelEquationT1, IV, args=(
                reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType))
            # result
            successStatus = True if len(sol) > 0 else False
            # all results
            # components, temperature layers
            dataYs = sol
        elif solverRootSet == "root":
            # root
            # lm, krylov, anderson, hybr, broyden1, linearmixing, diagbroyden, excitingmixing
            sol = optimize.root(HomoModelClass.modelEquationT1, IV, args=(
                reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType), method='anderson')
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x
        elif solverRootSet == "least_squares":
            sol = optimize.least_squares(HomoModelClass.modelEquationT1, IV, bounds=setBounds, args=(
                reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType))
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x
        elif solverRootSet == "minimize":
            sol = optimize.minimize(HomoModelClass.modelEquationT1, IV, args=(
                reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType), bounds=setBounds)
            # result
            successStatus = sol.success
            # all results
            # components, temperature layers
            dataYs = sol.x

        # check
        if successStatus is False:
            raise

        # std format
        dataYs_Reshaped = np.reshape(
            dataYs, (noLayer, varNoRows, varNoColumns))
        # data
        # -> concentration
        dataYs_Concentration_DiLeVa = dataYs_Reshaped[:-1]
        # -> temperature
        dataYs_Temperature_DiLeVa = dataYs_Reshaped[-1] if processType != PROCESS_SETTING['ISO-THER'] else np.repeat(
            0, zNo).reshape((varNoRows, varNoColumns))

        # sort out
        params1 = (compNo, noLayer, varNoRows, varNoColumns)
        params2 = (Cif, Tf, processType)
        dataYs_Sorted = setOptimizeRootMethod(
            dataYs_Reshaped, params1, params2)
        # component concentration [kmol/m^3]
        dataYs_Concentration_ReVa = dataYs_Sorted['data1']
        # temperature [K]
        dataYs_Temperature_ReVa = dataYs_Sorted['data2']

        # REVIEW
        # convert concentration to mole fraction
        dataYs1_Ctot = np.sum(dataYs_Concentration_ReVa, axis=0)
        dataYs1_MoFri = dataYs_Concentration_ReVa/dataYs1_Ctot

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

        # FIXME
        # build matrix
        _dataYs = np.concatenate(
            (dataYs1_MoFri, dataYs_Temperature_ReVa), axis=0)

        # plot info
        plotTitle = f"Steady-State {processType} Pseudo-Homogeneous Modeling [T1], Computation Time: {elapsed} seconds"

        # check
        if successStatus is True:
            # plot setting: build (x,y) series
            XYList = pltc.plots2DSetXYList(dataXs, _dataYs)
            # -> add label
            dataList = pltc.plots2DSetDataList(XYList, labelList)
            # datalists
            dataLists = [dataList[0:compNo],
                         dataList[indexTemp]]
            # select datalist
            _dataListsSelected = selectFromListByIndex([0, 1], dataLists)
            # subplot result
            xLabelSet = "Reactor Length (m)"
            yLabelSet = "Concentration (mol/m^3)"
            pltc.plots2DSub(_dataListsSelected, xLabelSet,
                            yLabelSet, plotTitle)

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

    def modelEquationT1(y, reactionListSorted, reactionStochCoeff, FunParam, DimensionlessAnalysisParams, processType):
        """
            M8 model [steady-state modeling]
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
                        GaDii0: diffusivity coefficient [m^2/s]
                        GaThCoi0: gas thermal conductivity [J/s.m.K]
                        GaVii0: gas viscosity [Pa.s]
                        GaDe0: gas density [kg/m^3]
                        GaCpMeanMix0: heat capacity at constant pressure of gas mixture [kJ/kmol.K] | [J/mol.K]
                        GaThCoMix0: gas thermal conductivity [J/s.m.K]
                    meshSetting:
                        solverMesh: mesh installment
                        solverMeshSet: 
                            true: normal
                            false: mesh refinement
                        noLayerC: number of layers for concentration
                        noLayerT: number of layers for temperature
                        noLayer: number of layers
                        varNoLayer: var no in each layer
                        varNoLayerT: total number of vars (Ci,T,Cci,Tci)
                        varNoRows: number of var rows [j]
                        varNoColumns: number of var columns [i]
                        zNo: number of finite difference in z direction
                        rNo: number of orthogonal collocation points in r direction
                        dz: differential length [-]
                        dzs: differential length list [-]
                        zR: z ratio
                        zNoNo: number of nodes in the dense and normal sections
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
        # reactor length
        ReLe = ReSpec['ReLe']
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
        # feed temperature [K]
        Tf = DimensionlessAnalysisParams['Tf']
        # feed superficial velocity [m/s]
        vf = DimensionlessAnalysisParams['vf']
        # domain length [m]
        zf = DimensionlessAnalysisParams['zf']
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
        # Reynolds number
        ReNu0 = DimensionlessAnalysisParams['ReNu0']
        # Schmidt number
        ScNu0 = DimensionlessAnalysisParams['ScNu0']
        # Sherwood number
        ShNu0 = DimensionlessAnalysisParams['ShNu0']
        # Prandtl number
        PrNu0 = DimensionlessAnalysisParams['PrNu0']
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
        # reaction rate
        ri = np.zeros(compNo)
        ri0 = np.zeros(compNo)

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
        SpCo_mz = np.zeros((noLayerC, varNoRows, varNoColumns))
        # all species concentration in gas phase [kmol/m^3]
        SpCoi_z = np.zeros((compNo, zNo))
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

        # species concentration in gas phase [kmol/m^3]
        CoSpi = np.zeros(compNo)
        # dimensionless analysis
        CoSpi_ReVa = np.zeros(compNo)
        # total concentration [kmol/m^3]
        CoSp = 0
        # species concentration in solid phase (catalyst) [kmol/m^3]
        # shape
        CosSpiMatShape = (1, compNo)
        CosSpi_r = np.zeros(CosSpiMatShape)
        # dimensionless analysis
        CosSpi_r_ReVa = np.zeros(CosSpiMatShape)

        # flux
        MoFli_z = np.zeros(compNo)

        # NOTE
        # temperature [K]
        T_mz = np.zeros((varNoRows, varNoColumns))
        T_mz = yLoop[noLayer -
                     1] if processType != "iso-thermal" else np.repeat(0, zNo).reshape((varNoRows, varNoColumns))
        # temperature in the gas phase
        T_z = np.zeros(zNo)
        T_z = T_mz[0, :]
        # temperature in solid phase
        Ts_z = np.zeros((1, zNo))
        Ts_z = T_mz[1:]

        # NOTE
        ### dimensionless analysis ###

        # diff/dt
        # dxdt = []
        # matrix
        # dxdz Matrix
        dxdzMatShape = (noLayer, varNoRows, varNoColumns)
        dxdzMat = np.zeros(dxdzMatShape)

        # NOTE
        # define ode equations for each finite difference [zNo]
        for z in range(zNo):
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

            # temperature [K]
            T = T_z[z]
            T_ReVa = rmtUtil.calRealDiLessValue(T, Tf, "TEMP")

            # pressure [Pa]
            P = P_z[z]

            # FIXME
            # velocity
            # dimensionless value
            v = 1  # v_z[z]

            ## calculate ##
            # mole fraction in the gas phase [0,1]
            MoFri = np.array(
                rmtUtil.moleFractionFromConcentrationSpecies(CoSpi_ReVa))

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
            VoFlRai = calVolumetricFlowrateIG(P, T_ReVa, MoFlRai_Con1)

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
            # viscosity in the gas phase [Pa.s] | [kg/m.s]
            GaVii = GaVii0 if MODEL_SETTING['GaVii'] == "FIX" else calTest()
            # mixture viscosity in the gas phase [Pa.s] | [kg/m.s]
            # FIXME
            GaViMix = 2.5e-5  # f(yi,GaVi,MWs);
            # kinematic viscosity in the gas phase [m^2/s]
            GaKiViMix = GaViMix/GaDe

            # REVIEW
            # thermal conductivity - gas phase [J/s.m.K]
            GaThCoi = GaThCoi0 if MODEL_SETTING['GaThCoi'] == "FIX" else calTest(
            )
            # dimensionless
            GaThCoi_DiLe = GaThCoi/GaThCoi0
            # mixture thermal conductivity - gas phase [J/s.m.K]
            # FIXME
            # convert
            GaThCoMix = GaThCoMix0
            # dimensionless analysis
            GaThCoMix_DiLeVa = 1
            # effective thermal conductivity - gas phase [J/s.m.K]
            GaThCoEff = BeVoFr*GaThCoMix
            # dimensionless analysis
            GaThCoEff_DiLeVa = GaThCoMix_DiLeVa  # BeVoFr*GaThCoMix_DiLeVa

            # REVIEW
            # diffusivity coefficient - gas phase [m^2/s]
            GaDii = GaDii0 if MODEL_SETTING['GaDii'] == "FIX" else calTest()
            # dimensionless analysis
            GaDii_DiLeVa = GaDii/GaDii0
            # effective diffusivity coefficient - gas phase
            GaDiiEff = GaDii*BeVoFr
            # dimensionless analysis
            GaDiiEff_DiLeVa = GaDiiEff/GaDii0
            # effective diffusivity - solid phase [m2/s]
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
            loopVars0 = (T_ReVa, P, MoFri, CoSpi_ReVa)

            # component formation rate [mol/m^3.s]
            # check unit
            RiLoop = np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))
            Ri = np.copy(RiLoop)

            # REVIEW
            # component formation rate [kmol/m^3.s]
            riRes = componentFormationRate(
                compNo, comList, reactionStochCoeff, Ri)

            ri = riRes  # (1-BeVoFr)*riRes

            # overall formation rate [kmol/m^3.s]
            OvR = np.sum(ri)

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

            # enthalpy change from Tref to T [kJ/kmol] | [J/mol]
            # enthalpy change
            EnChList = np.array(
                calEnthalpyChangeOfReaction(reactionListSorted, T_ReVa))
            # heat of reaction at T [kJ/kmol] | [J/mol]
            HeReT = np.array(EnChList + StHeRe25)
            # overall heat of reaction [kJ/m^3.s]
            OvHeReT = np.dot(Ri, HeReT)

            # FIXME
            # effective heat capacity - solid phase [kJ/m^3.K]
            SoCpMeanMixEff = CoSp*GaCpMeanMix*CaPo + (1-CaPo)*CaDe*CaSpHeCa

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
            # bulk temperature [K]
            T_c = T_z[z]

            # velocity from global concentration
            # check BC
            # if z == 0:
            #     # BC1
            #     constT_BC1 = 2*dz*zRef*(MoFl*GaCpMeanMix/1000)
            #     # forward
            #     T_f = T_z[z+1]
            #     # backward
            #     T_b = constT_BC1*(T0 - T_c) + GaThCoEff*T_f
            # elif z == zNo - 1:
            #     # BC2
            #     # backward
            #     T_b = T_z[z - 1]
            #     # forward
            #     T_f = T_b
            # else:
            #     # interior nodes
            #     # backward
            #     T_b = T_z[z-1]
            #     # forward
            #     T_f = T_z[z+1]

            # dxdt_v_T = (T_z[z] - T_b)/dz
            # CoSp x 1000
            # OvR x 1000
            # dxdt_v = (1/(CoSp*1000))*((-SuGaVe/CONST.R_CONST) *
            #                           ((1/T_z[z])*dxdt_P - (P_z[z]/T_z[z]**2)*dxdt_v_T) - ToMaTrBeGaSo_z*1000)
            # velocity [forward value] is updated
            # backward value of temp is taken
            # dT/dt will update the old value
            # v_z[z+1] = dxdt_v*dz + v_z[z]
            # FIXME
            v_z[z+1] = v_z[z]
            # dimensionless analysis
            v_z_DiLeVa = rmtUtil.calDiLessValue(v_z[z+1], vf)

            # NOTE
            # diff/dz
            # dxdt = []

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

                # convective term -> [no unit]
                _convectiveTerm = -1*v_z_DiLeVa*dCdz
                # dispersion term [kmol/m^3.s] -> [no unit]
                _dispersionFluxC = (BeVoFr*GaDii_DiLeVa[i]/PeNuMa0[i])*d2Cdz2
                # reaction term [kmol/m^3.s] -> [no unit]
                _reactionTerm = (1/GaMaCoTe0[i])*ri[i]
                # mass balance
                # convective, dispersion, reaction terms
                dxdz_C = (_convectiveTerm + _dispersionFluxC + _reactionTerm)
                dxdzMat[i][0][z] = dxdz_C

            # NOTE
            # energy balance (temperature) [K]
            # temp [K]
            # T_c = T_z[z]

            if processType != PROCESS_SETTING['ISO-THER']:
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

                # NOTE
                # convective term
                _convectiveTerm = -1*v_z_DiLeVa*GaDe_DiLeVa*GaCpMeanMix_DiLeVa*dTdz
                # dispersion flux [kJ/m^3.s] -> [no unit]
                _dispersionFluxT = (1/PeNuHe0)*GaThCoEff_DiLeVa*d2Tdz2
                # heat of reaction term
                # OvHeReT [kJ/m^3.s] -> [no unit]
                OvHeReT_Conv = -1*OvHeReT
                _reactionHeatTerm = (1/GaHeCoTe0)*OvHeReT_Conv
                # heat exchange term [kJ/m^3.s] -> [no unit]
                _heatExchangeTerm = (1/GaHeCoTe0)*Qm
                # convective flux, diffusive flux, enthalpy of reaction, heat exchange term
                dxdt_T = (_convectiveTerm + _dispersionFluxT +
                          _reactionHeatTerm + _heatExchangeTerm)
                dxdzMat[indexT][0][z] = dxdt_T

        # NOTE
        # flat array
        dxdt = dxdzMat.flatten().tolist()

        return dxdt
