# BATCH REACTOR
# --------------

# import packages/modules
import math as MATH
import numpy as np
from PyREMOT.library import plotClass as pltc


class batchReactorClass:
    """
    batch reactor class
    assumptions:
        ideal gas, 
        ideal and incompressible liquid
        homogeneous reaction/single phase
        perfect mixing: no spatial variation in temperature and composition 
    """

    def __init__(self, modelInput, internalData, reactionListSorted):
        self.modelInput = modelInput
        self.internalData = internalData
        self.reactionListSorted = reactionListSorted

    def runM3(self):
        """
        constant-volume batch reactor
        definitions:
            the volume of reaction mixture does not change with time
        """

        # operating conditions
        P = self.modelInput['operating-conditions']['pressure']
        T = self.modelInput['operating-conditions']['temperature']

        # component list
        compList = self.modelInput['feed']['components']['shell']
        labelList = compList.copy()
        labelList.append("Temperature")

        # initial values
        # -> mole fraction
        Coi = self.modelInput['feed']['concentration']
        IV = []
        IV.extend(Coi)
        # print(f"IV: {IV}")

        # time span
        # t = (0.0, rea_L)
        # t = np.array([0, rea_L])
        # times = np.linspace(t[0], t[1], 20)
        # tSpan = np.linspace(0, rea_L, 25)

        # ode call
        # sol = solve_ivp(PackedBedReactorClass.modelEquationM1,
        #                 t, IV, method="LSODA", t_eval=times, args=(P, T))
        sol = 1

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

    def modelEquationM3(t, y, P):
        """
            M1 model
            mass balance equations
            modelParameters:
                pressure [Pa]
        """
        # t
        # print(f"t: {t}")
        # components no
        # y: component mole fraction, temperature
        compNo = len(y[:-1])
        indexT = compNo

        # concentration list
        Coi = y[:-2]

        # temperature [K]
        T = y[indexT]

        # kinetics
        Ri = 1
        # batchReactorClass.modelReactions(P, T, Coi)
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
        A1 = 1/1
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
