# GENERAL EQUATIONS
# ------------------


# import packages/modules
import numpy as np
import re
from typing import List
from core import constants as CONST
# add err class

#


class rmtUtilityClass:
    #
    def __init__(self):
        pass

    @staticmethod
    def extractCompData(compData, compProperty):
        """
        build a list of desired component data
        args:
            compData: component database dict list
            compProperty: property name
        """
        # try/except
        try:
            # prop list
            propList = [item[compProperty] for item in compData]
            return propList
        except Exception as e:
            raise

    @staticmethod
    def extractSingleCompData(compId, compData, compProperty):
        """
        desired component data
        args:
            compId: component name, such as CO2, ...
            compData: component database dict list
            compProperty: property name
        """
        # try/except
        try:
            # prop list
            propList = [item[compProperty]
                        for item in compData if item['symbol'] == compId]
            # check
            if len(propList) == 0:
                raise
            else:
                return propList[0]
        except Exception as e:
            raise

    @staticmethod
    def mixtureMolecularWeight(MoFri, MWi, unit="g/mol"):
        """
        calculate mixture molecular weight [g/mol]
        args:
            MoFri: component mole fraction
            MWi: molecular weight [g/mol]
        """
        # try/exception
        try:
            # check
            if not isinstance(MoFri, np.ndarray):
                if isinstance(MoFri, List):
                    MoFri0 = np.array(MoFri)
            else:
                MoFri0 = np.copy(MoFri)

            if not isinstance(MWi, np.ndarray):
                if isinstance(MWi, List):
                    MWi0 = np.array(MWi)
            else:
                MWi0 = np.copy(MWi)

            # check
            if MoFri0.size != MWi0.size:
                raise Exception("elements are not equal")
            #
            MixMoWe = np.dot(MoFri0, MWi0)

            # check unit
            if unit == 'g/mol':
                MixMoWe
            elif unit == 'kg/mol':
                MixMoWe = MixMoWe*1e-3
            elif unit == 'kg/kmol':
                MixMoWe = MixMoWe

            return MixMoWe
        except Exception as e:
            print(e)

    @staticmethod
    def volumetricFlowrateSTP(VoFlRa, P, T):
        """
        calculate volumetric flowrate at STP conditions
        args:
            VoFlRa []
            P: pressure [Pa]
            T: temperature [K]
            VoFlRaSTP []
        """
        VoFlRaSTP = VoFlRa*(P/CONST.Pstp)*(CONST.Tstp/T)
        return VoFlRaSTP

    @staticmethod
    def VoFlRaSTPToMoFl(VoFlRaSTP):
        """
        convert volumetric flowrate [stp] to molar flowrate [ideal gas]
        args:
            VoFlRaSTP [m^3/s]
            MoFlRaIG [mol/s]
        """
        MoFlRaIG = (VoFlRaSTP/0.02241)
        return MoFlRaIG

    @staticmethod
    def reactorCrossSectionArea(BePo, ReDi):
        """
            calculate reactor cross section area
            BePo: bed porosity [-]
            ReDi: reactor diameter [m]
            ReCrSeAr [m^2]
        """
        ReCrSeAr = BePo*(CONST.PI_CONST*(ReDi**2)/4)
        return ReCrSeAr

    @staticmethod
    def componentInfo(compList):
        """
            return component no. and name (symbol)
            args:
                compList: list of components' symbols
        """
        # try/except
        try:
            # component no
            if isinstance(compList, List):
                compNo = len(compList)
            else:
                raise

            # res
            res = {
                "compNo": compNo,
                "compList": compList
            }

            return res
        except Exception as e:
            raise

    @staticmethod
    def buildReactionList(reactionDict):
        """
            return reaction list [string]
            args:
                reactionDict: reaction dict
        """
        # try/except
        try:
            dictVal = reactionDict.values()
            return list(dictVal)
        except Exception as e:
            raise

    @staticmethod
    def buildReactionCoefficient(reactionDict):
        """
            build reaction coefficient
            args:
                reactionDict: reaction dict
        """
        # try/except
        try:
            # reaction list
            reactionList = rmtUtilityClass.buildReactionList(reactionDict)
            # print(f"reaction list: {reactionList}")

            # sorted reaction list
            reactionListSorted = []

            for reaction in reactionList:
                # analyze reactions
                reaType = reaction.replace("<", "").replace(">", "")
                # reactant/products list
                compR = reaType.replace(r" ", "").split("=")
                # print(f"compR1 {compR}")

                # componets
                reactantList = re.findall(
                    r"([0-9.]*)([a-zA-Z0-9.]+)", compR[0])
                # print(f"reactantList {reactantList}")
                productList = re.findall(r"([0-9.]*)([a-zA-Z0-9.]+)", compR[1])
                # print(f"productList {productList}")
                # print("------------------")

                # reactant coefficient
                _loop1 = [{"symbol": i[1], "coeff": -1*float(i[0])} if len(i[0]) != 0 else {"symbol": i[1], "coeff": -1.0}
                          for i in reactantList]
                # product coefficient
                _loop2 = [{"symbol": i[1], "coeff": float(i[0])} if len(i[0]) != 0 else {"symbol": i[1], "coeff": 1.0}
                          for i in productList]
                # store dict
                _loop3 = {
                    "reactants": _loop1,
                    "products": _loop2
                }
                # store
                reactionListSorted.append(_loop3)

            # print("reactionListSorted: ", reactionListSorted)
            # res
            return reactionListSorted
        except Exception as e:
            raise

    @staticmethod
    def buildReactionCoeffVector(reactionListSorted):
        """
            build reaction coeff vector
        """
        # try/except
        try:
            # build list
            reactionCoeff = []
            #
            for element in reactionListSorted:
                # reactant coefficient
                _loop1 = [[item['symbol'], float(item['coeff'])]
                          for item in element['reactants']]
                # product coefficient
                _loop2 = [[item['symbol'], float(item['coeff'])]
                          for item in element['products']]
                # vector
                _loop3 = []
                _loop3.extend(_loop1)
                _loop3.extend(_loop2)
                _loop4 = _loop3.copy()
                reactionCoeff.append(_loop4)

            # res
            return reactionCoeff
        except Exception as e:
            raise

    @ staticmethod
    def buildreactionRateExpr(reactionRateExprDict):
        """
        build a list of reaction rate expression
        args:
            reactionRateExprDict: a dictionary contains all reaction rate expr
            {"R1": "r1(C,T,y), ...}
        """
        # try/except
        try:
            print(0)
        except Exception as e:
            raise

    @ staticmethod
    def moleFractionFromConcentrationSpecies(CoSpi):
        """
        calculate: mole fraction
        args:
            CoSpi: concentration species [mol/m^3]
        """
        # try/except
        try:
            MoFri = CoSpi/np.sum(CoSpi)
            return MoFri
        except Exception as e:
            raise

    @ staticmethod
    def moleFractionFromConcentrationSpeciesMat(CoSpi):
        """
        calculate: mole fraction from matrix
            mat[rNO, compNo]
        args:
            CoSpi: concentration species [mol/m^3] | [kmol/m^3]
        """
        # try/except
        try:
            # size
            _shape = np.shape(CoSpi)
            _SpCoT = np.sum(CoSpi, axis=1)
            _SpCoT_Reshape = _SpCoT.reshape((_shape[0], 1))
            MoFri = CoSpi/_SpCoT_Reshape
            return MoFri
        except Exception as e:
            raise

    @ staticmethod
    def moleFractionFromSpeciesMolarFlowRate(MoFlRai):
        """
        calculate: mole fraction
        args:
            MoFlRai: species molar flowrate [mol/s]
        """
        # try/except
        try:
            MoFri = MoFlRai/np.sum(MoFlRai)
            return MoFri
        except Exception as e:
            raise

    @staticmethod
    def buildComponentList(componentDataDict):
        """ 
        build component list participated in the reaction
        this list is used for component availability in the app database
        """
        # try/except
        try:
            # all
            compList0 = []
            # shell
            shellComp = componentDataDict['shell']
            if shellComp:
                compList0.extend(shellComp)
            # tube
            tubeComp = componentDataDict['tube']
            if tubeComp:
                compList0.extend(tubeComp)
            # medium
            mediumComp = componentDataDict['medium']
            if mediumComp:
                compList0.extend(mediumComp)
            # all
            compList = list(dict.fromkeys(compList0))
            return compList

        except Exception as e:
            raise

    @staticmethod
    def calSuperficialGasVelocityFromEOS(MoFl, P, T):
        """ 
        calculate: superficial gas velocity from EOS ideal gas [m/s]
        args:
            MoFl: molar flux [mol/m^2.s]
            P: pressure [Pa]
            T: temperature [K]
        """
        # try/except
        try:
            # superficial gas velocity [m/s]
            SuGaVe = MoFl*T*CONST.R_CONST/P
            return SuGaVe
        except Exception as e:
            pass

    @staticmethod
    def calEquivalentParticleDiameter(data, type="sphere"):
        """ 
        calculate: equivalent particle diameter [m]
        args: 
            data: 
                sphere: 
                    data['D']: sphere diameter [m]
        """
        # try/except
        try:
            if type == "sphere":
                ds = data['R']
                rs = ds/2
                # surface area
                A = 4*CONST.PI_CONST*(rs**2)
                # volume
                V = (4/3)*CONST.PI_CONST*(rs**3)
            elif type == "cylinder":
                pass
            # ratio of particle surface area
            av = A/V
            # REVIEW
            # equivalent particle diameter
            EqPaDi = 6/av
            return EqPaDi
        except Exception as e:
            raise

    @staticmethod
    def calSpPaSuArToFrFl(PaDe, BeVoFr):
        """ 
        calculate: specific particle surface area to the free fluid [m^2 of partiles/m^3 of fluid]
        args:
            PaDe: particle diameter [m]
            BeVoFr: bed void fraction 
        """
        # try/exception
        try:
            # specific surface area of particle to the free fluid
            SpPaSuArToFrFl = 3*(1-BeVoFr)/(PaDe/2)
            return SpPaSuArToFrFl
        except Exception as e:
            raise

    @staticmethod
    def calGaVeFromEOS(GaVef, Ctotf, Ctot, Pf, P):
        '''
        calculate: gas velocity change due to mole change and pressure drop
        args:
            GaVef: inlet feed velocity [m/s]
            Ctotf: inlet feed total concentration [kmol/m^3]
            Ctot: total concentration [kmol/m^3]
            Pf: inlet feed pressure [Pa]
            P: pressure [Pa]
        '''
        # try/exception
        try:
            # gas velocity [m/s]
            GaVe = GaVef*(Ctot/Ctotf)*(Pf/P)
            return GaVe
        except Exception as e:
            raise

    @staticmethod
    def calHeatExchangeBetweenReactorMedium(Tm, T, U, a, unit="J/m^3.s"):
        '''
        calculate: heat exchange between reactor and sourrounding 
            cooling for exothermic reactions
            warming for endothermic reaction
        args:
            Tm: medium temperature [K]
            T: fluid temperature [K]
            U: overall heat transfer coefficient between fluid, reactor wall, sourounding [J/m^2.s.K]
            a: effective heat transfer area per unit of reactor volume [m^2/m^3]
        '''
        # try/exception
        try:
            if Tm == 0:
                # adiabatic
                Qm = 0
            else:
                # heat added/removed from the reactor
                # Tm > T: heat is added (positive sign)
                # T > Tm: heat removed (negative sign)
                Qm = U*a*(Tm - T)

            # check unit
            if unit == 'kJ/m^3.s' and Qm != 0:
                Qm = Qm*1e-3

            return Qm
        except Exception as e:
            raise

    @staticmethod
    def calRealDiLessValue(xr, xf, mode="G"):
        """
        calculate real value of scaled variable
        args:
            xr: dimensionless var
            xf: initial value var
            mode: 
                G: general
                T: temperature
        output:
            x: real value var
        """
        # try/except
        try:
            if mode == "TEMP":
                return xr*xf + xf
            else:
                return xr*xf
        except Exception as e:
            raise

    @staticmethod
    def calDiLessValue(x, xf, mode="G"):
        """
        calculate value of dimensionless var [0,1]
        args:
            xr: real value var 
            xf: initial value var
            mode: 
                G: general
                T: temperature
        output:
            xr: dimensionless var
        """
        # try/except
        try:
            if mode == "TEMP":
                return (x-xf)/xf
            else:
                return x/xf
        except Exception as e:
            raise
