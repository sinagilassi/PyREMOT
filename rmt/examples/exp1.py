# FINITE DIFFERENCE METHOD
# -------------------------

# import module/packages
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
# internal
from docs.rmtCore import rmtCoreClass
from docs.rmtReaction import reactionRateExe, componentFormationRate
from solvers.solFiDi import FiDiBuildCMatrix, FiDiBuildTMatrix
from docs.rmtUtility import rmtUtilityClass as rmtUtil
from reactionList import *
from docs.rmtThermo import *
from data1 import *

# ode no
# 3: concentration
odeNo = 6
# number of finite difference points
N = 3

# name of variables
comList = ["H2", "CO2", "H2O", "CO", "CH3OH", "DME"]
compNo = 6

# bulk concentration [kmol/m^3]
cti_H2 = 0.574947645267422
cti_CO2 = 0.287473822633711
cti_H2O = 1.14989529053484e-05
cti_CO = 0.287473822633711
cti_CH3OH = 1.14989529053484e-05
cti_DME = 1.14989529053484e-05
Cbs = np.array([cti_H2, cti_CO2, cti_H2O, cti_CO, cti_CH3OH, cti_DME])

# temperature in the fluid phase - bulk [K]
TBulk = 523

# diffusivity coefficient [m^2/s]
Dim = np.array([2.27347635942262e-06, 9.16831604900657e-07, 5.70318666607403e-07,
                9.98820628335698e-07, 5.40381353373092e-07, 4.28676364755756e-07])
# mass transfer coefficient [m/s]
km = np.array([0.0273301866548795,	0.0149179341780856,	0.0108707796723462,
               0.0157945517381349,	0.0104869502041277,	0.00898673624257253])

# reaction term Ri [kmol/m^3.s]
Ri_r = np.zeros((N, compNo))

# NOTE
### reactor ###
# catalyst particle diameter [m]
cat_d = 0.002
PaRa = cat_d/2
# catalyst porosity
CaPo = 0.78

# matrix structure
# rhs
rhsMatrixShape = odeNo*N
# fdydt
fdydtShape = odeNo*N

# SoCpMeanMixEff [kJ/m^3.K]
SoCpMeanMixEff = 279.34480838441203

# catalyst
const_Cs1 = 1/(CaPo*(PaRa**2))
const_Ts1 = 1/(SoCpMeanMixEff*(PaRa**2))


def fdydt(t, y, params):
    # parameters
    P_z, reactionStochCoeff, reactionListSorted, StHeRe25, SoThCoEff, HeTrCo, OvHeReT = params

    fxMat = np.zeros((odeNo, N))

    # reshape yj
    yj = np.array(y)
    yj_Reshape = np.reshape(yj, (odeNo, N))

    # concentration [kmol/m^3]
    SpCoi_r = yj_Reshape[0:-1, :]
    # temperature [K]
    Ts_r = yj_Reshape[-1, :]

    # component mole fraction
    # mole fraction in the solid phase
    # MoFrsi_r0 = CosSpi_r/CosSp_r
    MoFrsi_r = rmtUtil.moleFractionFromConcentrationSpeciesMat(
        SpCoi_r)

    # NOTE
    ## kinetics ##
    # solid phase
    ri_r = np.zeros((N, compNo))
    SoCpMeanMix = np.zeros(N)

    # net reaction rate expression [kmol/m^3.s]
    # rf[kmol/kgcat.s]*CaDe[kgcat/m^3]
    for r in range(N):
        #
        # r0 = np.array(PackedBedReactorClass.modelReactions(
        #     P_z[z], Ts_r[r], MoFrsi_r[r], CaDe))

        # loop
        loopVars0 = (Ts_r[r], P_z, MoFrsi_r[r], SpCoi_r[r])

        # component formation rate [mol/m^3.s]
        # check unit
        r0 = np.array(reactionRateExe(
            loopVars0, varis0, rates0))

        Ri_r[r, :] = r0

        # reset
        _riLoop = 0

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
    for k in range(odeNo):

        for i in range(N):
            if k < odeNo - 1:
                # loop
                _dCsdtiVarLoop = (Dim[i], km[i], Ri_r[i, :], Cbs[i])

                # dC/dt list
                dCsdti = FiDiBuildCMatrix(
                    compNo, PaRa, N, yj_Reshape[k, :], _dCsdtiVarLoop)

                for r in range(N):
                    # update
                    fxMat[i][r] = const_Cs1*dCsdti[r]

            else:
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
                    SoThCoEff_Conv, HeTrCo_Conv, OvHeReT_Conv, TBulk)

                # dTs/dt list
                dTsdti = FiDiBuildTMatrix(
                    compNo, PaRa, N, yj_Reshape[k, :], _dTsdtiVarLoop)

                for r in range(N):
                    # update
                    fxMat[odeNo-1][r] = const_Ts1*dTsdti[r]

    # NOTE
    # flat
    dxdt = fxMat.flatten().tolist()

    # return
    return dxdt


# NOTE
# class
rmtCoreClassSet = rmtCoreClass()


# init
reactionList = 1


# standard heat of reaction at 25C [kJ/kmol]
StHeRe25 = np.array(
    list(map(calStandardEnthalpyOfReaction, reactionList)))


# System of ODEs
# ----------------------------------------


# NOTE
# updated through each loop
# initial conditions (IC)

# initial values at t = 0 and z >> 0
noLayer = odeNo
varNoColumns = N

IVMatrixShape = (noLayer, N)
IV2D = np.zeros(IVMatrixShape)
# initialize IV2D
# -> concentration [kmol/m^3]
for m in range(noLayer - 1):
    for i in range(varNoColumns):
        # solid phase
        IV2D[m][i] = Cbs[m]


# temperature
for i in range(varNoColumns):
    IV2D[noLayer - 1][i] = TBulk


# flatten IV
IV = IV2D.flatten()


# time points [s]
tFinal = 5
t = (0.0, tFinal)
ts = 5
tSpan = np.linspace(0, tFinal, ts)

# solve equation
# parameters
params = {

}
# solve ODE
sol = solve_ivp(fdydt, t, IV, method="LSODA", t_eval=tSpan, args=(params,))
# y [ode(i),t]
solY = sol.y
# y [y(Xc(i)),Xc(i)]
solY = np.transpose(sol.y)


# separate res [C,T]
# solYC = solY[0:N*(odeNo-1), :]
# solYT = solY[N*(odeNo-1):N*odeNo, :]

# NOTE
# find value for x = 0: dy/dx = 0
ds = []
solPoint = []

for i in range(ts):
    # all res
    yiLoopAll = solY[i, :]

    # save result at time (s)
    solPoint.append([])

    # component res
    for k in range(compNo):
        # yi
        yisLoop = yiLoopAll
        # xi
        xisLoop = N
        # save loop

        solPoint[i].append({
            "xi": xisLoop,
            "yi": yisLoop
        })

# plot time step
tsStep = [2, 4]
# plot results
for i in tsStep:
    # component res
    for k in range(compNo):
        plt.plot(solPoint[i][k]['xi'], solPoint[i][k]['yi'], label="t=" +
                 str(i) + "Symbol = " + str(comList[k]))
