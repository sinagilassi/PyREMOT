# PACKED-BED REACTOR MODEL
# -------------------------

# import packages/modules
from core.errors import errGeneralClass as errGeneral
import numpy as np
from data.inputDataReactor import *
from scipy.integrate import solve_ivp
from core import constants as CONST


class PackedBedReactorClass:
    # def main():
    """
        Packed-bed Reactor Model
    """

    def __init__(self, modelInput):
        self.modelInput = modelInput

    def runM1(self):
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
        t = (0.0, rea_L)
        # tSpan = np.linspace(0, rea_L, 25)

        # ode call
        sol = solve_ivp(PackedBedReactorClass.modelEquationM1,
                        t, IV, args=(P, T))

        # res
        x = sol.y
        print(x)

        res = x
        return res

    # model equations

    def modelEquationM1(t, y, P, T):
        """
            M1 model
            mass balance equations
            modelParameters: 
                pressure [Pa] 
                temperature [K]
        """
        # operating conditions
        # P = modelParameters['pressure']
        # T = modelParameters['temperature']
        # # components
        # comp = modelParameters['components']
        # # components no
        # compNo = len(comp)

        #! loop vars
        # MoFri = np.copy(y)
        yi_H2 = y[0]
        yi_CO2 = y[1]
        yi_H2O = y[2]
        yi_CO = y[3]
        yi_CH3OH = y[4]
        yi_DME = y[5]

        # molar flux [kmol/m^2.s]
        MoFl = y[6]

        # mole fraction list
        MoFri = [yi_H2, yi_CO2, yi_H2O, yi_CO, yi_CH3OH, yi_DME]

        # kinetics
        Ri = PackedBedReactorClass.modelReactions(P, T, MoFri)
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
        A1 = 1/MoFl
        B1 = 1

        #  H2
        dxdt_H2 = A1*(R_H2 - y[1]*R_T)
        #  CO2
        dxdt_CO2 = A1*(R_CO2 - y[2]*R_T)
        #  H2O
        dxdt_H2O = A1*(R_H2O - y[3]*R_T)
        #  CO
        dxdt_CO = A1*(R_CO - y[4]*R_T)
        #  CH3OH
        dxdt_CH3OH = A1*(R_CH3OH - y[5]*R_T)
        #  DME
        dxdt_DME = A1*(R_DME - y[6]*R_T)
        #  overall
        dxdt_T = B1*R_T
        # build diff/dt
        dxdt = [dxdt_H2, dxdt_CO2, dxdt_H2O,
                dxdt_CO, dxdt_CH3OH, dxdt_DME, dxdt_T]
        return dxdt

    def modelReactions(P, T, y):
        try:
            # pressure [Pa]
            # temperature [K]

            #  kinetic constant
            # DME production
            #  [kmol/kgcat.s.bar2]
            K1 = 35.45*np.exp(-1.7069e4/(CONST.R_CONST*T))*1
            #  [kmol/kgcat.s.bar]
            K2 = 7.3976*np.exp(-2.0436e4/(CONST.R_CONST*T))*1
            #  [kmol/kgcat.s.bar]
            K3 = 8.2894e4*np.exp(-5.2940e4/(CONST.R_CONST*T))*1
            # adsorption constant [1/bar]
            KH2 = 0.249*np.exp(3.4394e4/(CONST.R_CONST*T))
            KCO2 = 1.02e-7*np.exp(6.74e4/(CONST.R_CONST*T))
            KCO = 7.99e-7*np.exp(5.81e4/(CONST.R_CONST*T))
            #  equilibrium constant
            Ln_KP1 = 4213/T - 5.752*np.log(T) - 1.707e-3 * \
                T + 2.682e-6*(T ** 2) - 7.232e-10*(T ** 3) + 17.6
            KP1 = np.exp(Ln_KP1)
            log_KP2 = 2167/T - 0.5194 * \
                np.log10(T) + 1.037e-3*T - 2.331e-7*(T ** 2) - 1.2777
            KP2 = np.power(10, log_KP2)
            Ln_KP3 = 4019/T + 3.707*np.log(T) - 2.783e-3 * \
                T + 3.8e-7*(T ** 2) - 6.56e-4/(T ** 3) - 26.64
            KP3 = np.exp(Ln_KP3)
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
            ra2 = 1+KCO2*PCO2+KCO*PCO+np.sqrt(KH2*PH2)
            ra3 = (1/KP1)*((PH2O*PCH3OH)/(PCO2*(PH2 ** 3)))
            r1 = K1*(ra1/(ra2 ** 3))*(1-ra3)*bulk_rho
            ra4 = PH2O-(1/KP2)*((PCO2*PH2)/PCO)
            r2 = K2*(1/ra2)*ra4*bulk_rho
            ra5 = ((PCH3OH ** 2)/PH2O)-(PCH3OCH3/KP3)
            r3 = K3*ra5*bulk_rho

            # result
            r = [r1, r2, r3]
            return r
        except Exception as e:
            print(e)
