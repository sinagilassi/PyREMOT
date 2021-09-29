# DIFFUSION INTO CATALYST PARTICLE
# ---------------------------------

# import packages/modules
import math as MATH
import numpy as np
from numpy.lib import math
from library.plot import plotClass as pltc
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
# internal
from data.inputDataReactor import *
from core import constants as CONST
from solvers.solSetting import solverSetting
from docs.rmtReaction import reactionRateExe, componentFormationRate
from solvers.solFiDi import FiDiBuildCMatrix, FiDiBuildTMatrix
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from docs.rmtThermo import *


class ParticleModelClass:
    '''
    catalyst diffusion-reaction dynamic/steady-state models
    '''

    def __init__(self, modelInput, internalData, reactionListSorted, reactionStochCoeffList):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted
        self.reactionStochCoeffList = reactionStochCoeffList

    def runT1(self):
        """
        M7 modeling case
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
        IVMatrixShape = (compNo, rNo)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(compNo):
            for i in range(rNo):
                # solid phase
                # FIXME
                IV2D[m][i] = SpCoi0[m]

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

            },
            "reactionRateExpr": reactionRateExpr
        }

        # FIXME
        # concentration in the bulk phase [kmol/m^3]
        cti_H2 = 0.574947645267422
        cti_CO2 = 0.287473822633711
        cti_H2O = 1.14989529053484e-05
        cti_CO = 0.287473822633711
        cti_CH3OH = 1.14989529053484e-05
        cti_DME = 1.14989529053484e-05
        Cbs = np.array([cti_H2, cti_CO2, cti_H2O, cti_CO, cti_CH3OH, cti_DME])

        # temperature in the bulk phase [K]
        Tb = 523

        # diffusivity coefficient [m^2/s]
        GaDii = np.array([2.27347635942262e-06, 9.16831604900657e-07, 5.70318666607403e-07,
                          9.98820628335698e-07, 5.40381353373092e-07, 4.28676364755756e-07])
        # mass transfer coefficient [m/s]
        MaTrCo = np.array([0.0273301866548795,	0.0149179341780856,	0.0108707796723462,
                           0.0157945517381349,	0.0104869502041277,	0.00898673624257253])

        # SoCpMeanMixEff [kJ/m^3.K]
        SoCpMeanMixEff = 279.34480838441203

        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = 1731

        # particle params
        ParticleParams = {
            "SoCpMeanMixEff": SoCpMeanMixEff,
            "HeTrCo": HeTrCo,
            "MaTrCo": MaTrCo,
            "GaDii": GaDii,
            "Cbs": Cbs,
            "Tb": Tb
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
        dataPacktime = np.zeros((compNo, tNo, rNo))

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
            sol = solve_ivp(ParticleModelClass.modelEquationT1, t, IV, method=solverIVP,
                            t_eval=times,  args=(ReactionParams, FunParam, ParticleParams))

            # ode result
            successStatus = sol.success
            # check
            if successStatus is False:
                raise

            # time interval
            dataTime = sol.t
            # all results
            # components layers
            dataYs = sol.y

            # REVIEW
            # convert concentration to mole fraction
            # dataYs1_Ctot = np.sum(dataYs, axis=0)
            # dataYs1_MoFri = dataYs/dataYs1_Ctot

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYCons": dataYs[:, -1],
            })

            # for m in range(compNo):
            #     # var list
            #     dataPacktime[m][i, :] = dataPack[i]['dataYCons'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

            checkme = 0

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

    def modelEquationT1(t, y, ReactionParams, FunParam, ParticleParams):
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
        '''
        # parameters
        # reaction params
        reactionListSorted = ReactionParams['reactionListSorted']
        reactionStochCoeff = ReactionParams['reactionStochCoeff']
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

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # particle parameters
        SoCpMeanMixEff = ParticleParams['SoCpMeanMixEff']
        HeTrCo = ParticleParams['HeTrCo']
        MaTrCo = ParticleParams['MaTrCo']
        GaDii = ParticleParams['GaDii']
        Cbs = ParticleParams['Cbs']
        Tb = ParticleParams['Tb']

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
        Ts_r = T0*np.ones(rNo)

        # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
        SoThCoEff = CaThCo*((1 - CaPo)/CaTo)

        # NOTE
        ### yi manage ###
        fxMat = np.zeros((compNo, rNo))
        # reshape yj
        yj = np.array(y)
        yj_Reshape = np.reshape(yj, (compNo, rNo))

        # concentration [kmol/m^3]
        SpCoi_mr = yj_Reshape

        # concentration species in the solid phase [kmol/m^3]
        SpCoi_r = np.zeros((rNo, compNo))
        # display concentration list in each oc point (rNo)
        for i in range(compNo):
            for r in range(rNo):
                _CosSpi_z = SpCoi_mr[i][r]
                SpCoi_r[r][i] = max(_CosSpi_z, CONST.EPS_CONST)

        # component mole fraction
        # mole fraction in the solid phase
        # MoFrsi_r0 = CosSpi_r/CosSp_r
        MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
            SpCoi_r)

        # NOTE
        ## kinetics ##
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        SoCpMeanMix = np.zeros(rNo)
        # reaction term Ri [kmol/m^3.s]
        Ri_r = np.zeros((rNo, reactionListNo))
        # overall enthalpy
        OvHeReT = np.zeros(rNo)

        # catalyst
        const_Cs1 = 1/(CaPo*(PaRa**2))
        const_Ts1 = 1/(SoCpMeanMixEff*(PaRa**2))

        # net reaction rate expression [kmol/m^3.s]
        # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
        for r in range(rNo):
            # loop
            loopVars0 = (Ts_r[r], P_z, MoFrsi_r[r], SpCoi_r[r])

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

        # ode eq [dy/dt] numbers
        for i in range(compNo):
            # central
            Ci_c = Cbs[i]
            # species concentration at different points of particle radius [rNo]
            # [Cs[3], Cs[2], Cs[1], Cs[0]]
            _Cs_r = SpCoi_r[:, i].flatten()

            # loop
            _dCsdtiVarLoop = (GaDii[i], MaTrCo[i],
                              ri_r[:, i], Ci_c, CaPo)

            # dC/dt list
            dCsdti = FiDiBuildCMatrix(
                compNo, PaRa, rNo, _Cs_r, _dCsdtiVarLoop, mode="test")

            for r in range(rNo):
                # update
                fxMat[i][r] = const_Cs1*dCsdti[r]

        # NOTE
        # flat
        dxdt = fxMat.flatten().tolist()

        # print
        print(f"time: {t} seconds")

        # return
        return dxdt


# NOTE
# non-isothermal system

    def runT2(self):
        """
        M7 modeling case
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
        IVMatrixShape = (noLayer, rNo)
        IV2D = np.zeros(IVMatrixShape)
        # initialize IV2D
        # -> concentration [kmol/m^3]
        for m in range(noLayer - 1):
            for i in range(rNo):
                # solid phase
                # FIXME
                IV2D[m][i] = 1e-3  # SpCoi0[m]

        # temperature
        for i in range(rNo):
            IV2D[noLayer - 1][i] = T

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

            },
            "reactionRateExpr": reactionRateExpr
        }

        # FIXME
        # concentration in the bulk phase [kmol/m^3]
        cti_H2 = 0.574947645267422
        cti_CO2 = 0.287473822633711
        cti_H2O = 1.14989529053484e-05
        cti_CO = 0.287473822633711
        cti_CH3OH = 1.14989529053484e-05
        cti_DME = 1.14989529053484e-05
        Cbs = np.array([cti_H2, cti_CO2, cti_H2O, cti_CO, cti_CH3OH, cti_DME])

        # temperature in the bulk phase [K]
        Tb = 523

        # diffusivity coefficient [m^2/s]
        GaDii = np.array([2.27347635942262e-06, 9.16831604900657e-07, 5.70318666607403e-07,
                          9.98820628335698e-07, 5.40381353373092e-07, 4.28676364755756e-07])
        # mass transfer coefficient [m/s]
        MaTrCo = np.array([0.0273301866548795,	0.0149179341780856,	0.0108707796723462,
                           0.0157945517381349,	0.0104869502041277,	0.00898673624257253])

        # SoCpMeanMixEff [kJ/m^3.K]
        SoCpMeanMixEff = 279.34480838441203

        # heat transfer coefficient - gas/solid [J/m^2.s.K]
        HeTrCo = 1731

        # particle params
        ParticleParams = {
            "SoCpMeanMixEff": SoCpMeanMixEff,
            "HeTrCo": HeTrCo,
            "MaTrCo": MaTrCo,
            "GaDii": GaDii,
            "Cbs": Cbs,
            "Tb": Tb
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

            # ode call
            # method [1]: LSODA, [2]: BDF, [3]: Radau
            sol = solve_ivp(ParticleModelClass.modelEquationT2,
                            t, IV, method=solverIVP, t_eval=times,  args=(ReactionParams, FunParam, ParticleParams))

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
                dataYs[:, -1], (noLayer, rNo))

            # component concentration [kmol/m^3]
            dataYs1SolidPhase = dataYs_Reshaped[:-1]

            # REVIEW
            # convert concentration to mole fraction
            dataYs1_Ctot = np.sum(dataYs1SolidPhase, axis=0)
            dataYs1_MoFri = dataYs1SolidPhase/dataYs1_Ctot

            # temperature - 2d matrix
            dataYs2SolidPhase = dataYs_Reshaped[indexTemp]

            # combine
            _dataYs = np.concatenate(
                (dataYs1_MoFri, dataYs2SolidPhase), axis=0)

            # save data
            dataPack.append({
                "successStatus": successStatus,
                "dataTime": dataTime[-1],
                "dataYs": _dataYs,
                "dataYCons": dataYs1SolidPhase,
                "dataYTemps": dataYs2SolidPhase,
            })

            # for m in range(varNo):
            #     # var list
            #     dataPacktime[m][i, :] = dataPack[i]['dataYs'][m, :]

            # update initial values [IV]
            IV = dataYs[:, -1]

        # NOTE
        # end of computation
        end = timer()
        elapsed = roundNum(end - start)

    def modelEquationT2(t, y, ReactionParams, FunParam, ParticleParams):
        '''
        T2 model: non-isothermal system
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
        '''
        # parameters
        # reaction params
        reactionListSorted = ReactionParams['reactionListSorted']
        reactionStochCoeff = ReactionParams['reactionStochCoeff']
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

        # reaction rate expressions
        reactionRateExpr = FunParam['reactionRateExpr']
        # using equation
        varisSet = reactionRateExpr['VARS']
        ratesSet = reactionRateExpr['RATES']

        # particle parameters
        SoCpMeanMixEff = ParticleParams['SoCpMeanMixEff']
        HeTrCo = ParticleParams['HeTrCo']
        MaTrCo = ParticleParams['MaTrCo']
        GaDii = ParticleParams['GaDii']
        Cbs = ParticleParams['Cbs']
        Tb = ParticleParams['Tb']

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

        # SoThCoEff0 = CaPo*SoThCoMix + (1 - CaPo)*CaThCo
        SoThCoEff = CaThCo*((1 - CaPo)/CaTo)

        # NOTE
        ### yi manage ###
        fxMat = np.zeros((noLayer, rNo))
        # reshape yj
        yj = np.array(y)
        yj_Reshape = np.reshape(yj, (noLayer, rNo))

        # concentration [kmol/m^3]
        SpCoi_mr = yj_Reshape[0:-1, :]
        # temperature [K]
        Ts_r = yj_Reshape[-1, :]

        # concentration species in the solid phase [kmol/m^3]
        SpCoi_r = np.zeros((rNo, compNo))
        # display concentration list in each oc point (rNo)
        for i in range(compNo):
            for r in range(rNo):
                _CosSpi_z = SpCoi_mr[i][r]
                SpCoi_r[r][i] = max(_CosSpi_z, CONST.EPS_CONST)

        # component mole fraction
        # mole fraction in the solid phase
        # MoFrsi_r0 = CosSpi_r/CosSp_r
        MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
            SpCoi_r)

        # NOTE
        ## kinetics ##
        # solid phase
        ri_r = np.zeros((rNo, compNo))
        SoCpMeanMix = np.zeros(rNo)
        # reaction term Ri [kmol/m^3.s]
        Ri_r = np.zeros((rNo, reactionListNo))
        # overall enthalpy
        OvHeReT = np.zeros(rNo)

        # catalyst
        const_Cs1 = 1/(CaPo*(PaRa**2))
        const_Ts1 = 1/(SoCpMeanMixEff*(PaRa**2))

        # net reaction rate expression [kmol/m^3.s]
        # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
        for r in range(rNo):
            # loop
            loopVars0 = (Ts_r[r], P_z, MoFrsi_r[r], SpCoi_r[r])

            # component formation rate [mol/m^3.s]
            # check unit
            r0 = np.array(reactionRateExe(
                loopVars0, varisSet, ratesSet))

            Ri_r[r, :] = r0

            # REVIEW
            # component formation rate [kmol/m^3.s]
            ri_r[r] = componentFormationRate(
                compNo, comList, reactionStochCoeff, Ri_r[r])

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

        # ode eq [dy/dt] numbers
        for i in range(compNo):
            # central
            Ci_c = Cbs[i]
            # species concentration at different points of particle radius [rNo]
            # [Cs[3], Cs[2], Cs[1], Cs[0]]
            _Cs_r = SpCoi_r[:, i].flatten()

            # loop
            _dCsdtiVarLoop = (GaDii[i], MaTrCo[i],
                              ri_r[:, i], Ci_c, CaPo)

            # dC/dt list
            dCsdti = FiDiBuildCMatrix(
                compNo, PaRa, rNo, _Cs_r, _dCsdtiVarLoop, mode="test")

            for r in range(rNo):
                # update
                fxMat[i][r] = const_Cs1*dCsdti[r]

        # bulk temperature [K]
        T_c = Tb

        # temperature at different points of particle radius [rNo]
        # Ts[3], Ts[2], Ts[1], Ts[0]
        _Ts_r = Ts_r

        # dC/dt list
        # convert
        # [J/s.m.K] => [kJ/s.m.K]
        SoThCoEff_Conv = SoThCoEff/1000
        # OvHeReT [kJ/m^3.s]
        OvHeReT_Conv = -1*OvHeReT
        # HeTrCo [J/m^2.s.K] => [kJ/m^2.s.K]
        HeTrCo_Conv = HeTrCo/1000
        # var loop
        _dTsdtiVarLoop = (
            SoThCoEff_Conv, HeTrCo_Conv, OvHeReT_Conv, T_c, CaPo)

        # dTs/dt list
        dTsdti = FiDiBuildTMatrix(
            compNo, PaRa, rNo, _Ts_r, _dTsdtiVarLoop, mode="test")

        for r in range(rNo):
            # update
            fxMat[indexT][r] = const_Ts1*dTsdti[r]

        # NOTE
        # flat
        dxdt = fxMat.flatten().tolist()

        # print
        print(f"time: {t} seconds")

        # return
        return dxdt
