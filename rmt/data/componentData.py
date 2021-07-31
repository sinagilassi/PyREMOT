# COMPONENT PROPERTIES
# ---------------------

# import packages/modules
import numpy as np

# molecular weight [g/mol]
MW_H2 = 2.0
MW_CO2 = 44.01
MW_H2O = 18.01
MW_CO = 28.01
MW_CH3OH = 32.04
MW_DME = 46.07
MW_N2 = 28
MW_CH4 = 16.04
MW_C2H4 = 28.05
# ->
MWi = [MW_H2, MW_CO2, MW_H2O, MW_CO, MW_CH3OH, MW_DME]

# critical temperature [K]
Tc_CO2 = 304.12
Tc_H2 = 33.25
Tc_CH3OH = 512.64
Tc_H2O = 647.14
Tc_CO = 132.85
Tc_DME = 400
Tc_N2 = 126.192
Tc_CH4 = 190.56
Tc_C2H4 = 282.34

# critical pressure [bar]
Pc_CO2 = 73.74
Pc_H2 = 12.97
Pc_CH3OH = 80.97
Pc_H2O = 220.64
Pc_CO = 34.94
Pc_DME = 53
Pc_N2 = 33.98
Pc_CH4 = 45.99
Pc_C2H4 = 50.41

# acentric factor
w_CO2 = 0.239
w_H2 = -0.216
w_CH3OH = 0.556
w_H2O = 0.344
w_CO = 0.066
w_DME = 0.200
w_N2 = 0.039
w_CH4 = 0.011
w_C2H4 = 0.087

# heat of formation at 25C [kJ/mol]
dHf25_CO2 = -393.51
dHf25_H2 = 0.0
dHf25_CH3OH = -200.7
dHf25_H2O = -241.820
dHf25_CO = -110.53
dHf25_DME = -184.1
dHf25_N2 = 0
dHf25_O2 = 0
dHf25_NH3 = -46.22
dHf25_CH4 = -74.90
dHf25_C2H6 = -84.72
dHf25_C2H4 = 52.32
dHf25_C3H6 = 20.4
dHf25_C3H8 = -103.9
dHf25_C4H10 = -126.2
dHf25_C2H6O = -235.0

# standard gibbs free energy at 25 C [kJ/mol]
dGf25_CO2 = -394.6
dGf25_H2 = 0.0
dGf25_CH3OH = -162.6
dGf25_H2O = -228.7
dGf25_CO = -137.4
dGf25_DME = -0
dGf25_N2 = 0
dGf25_O2 = 0
dGf25_NH3 = -16.60
dGf25_CH4 = -50.83
dGf25_C2H6 = -32.90
dGf25_C2H4 = 68.17
dGf25_C3H6 = 62.76
dGf25_C3H8 = -23.50
dGf25_C4H10 = -17.2
dGf25_C2H6O = -168.4


# component database
componentDataStore = {
    "payload": [
        {
            "symbol": "CO2",
            "MW": MW_CO2,
            "Pc": Pc_CO2,
            "Tc": Tc_CO2,
            "w": w_CO2,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "22.243 + 5.98E-02*T + -3.50E-05*(T**2) + 7.46E-09*(T**3)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_CO2
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_CO2
            },
            "viscosity": ["eq1GasViscosity"]
        },
        {
            "symbol": "H2",
            "MW": MW_H2,
            "Pc": Pc_H2,
            "Tc": Tc_H2,
            "w": w_H2,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "26.879 + 4.35E-03*T + -3.30E-07*(T**2)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_H2
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_H2
            }
        },
        {
            "symbol": "CH3OH",
            "MW": MW_CH3OH,
            "Pc": Pc_CH3OH,
            "Tc": Tc_CH3OH,
            "w": w_CH3OH,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "19.038 + 9.15E-02*T + -1.22E-05*(T**2) + -8.03E-09*(T**3)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_CH3OH
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_CH3OH
            }
        },
        {
            "symbol": "H2O",
            "MW": MW_H2O,
            "Pc": Pc_H2O,
            "Tc": Tc_H2O,
            "w": w_H2O,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "29.163 + 1.45E-02*T + -2.02E-06*(T**2)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_H2O
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_H2O
            }
        },
        {
            "symbol": "CO",
            "MW": MW_CO,
            "Pc": Pc_CO,
            "Tc": Tc_CO,
            "w": w_CO,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "27.113 + 6.55E-03*T + -1.00E-06*(T**2)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_CO
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_CO
            }
        },
        {
            "symbol": "DME",
            "MW": MW_DME,
            "Pc": Pc_DME,
            "Tc": Tc_DME,
            "w": w_DME,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "19.8 + 0.17*T + -5.66e-5*(T**2)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_DME
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_DME
            }
        },
        {
            "symbol": "N2",
            "MW": MW_N2,
            "Pc": Pc_N2,
            "Tc": Tc_N2,
            "w": w_N2,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "28.883 + -1.57E-03*T + 8.08E-06*(T**2) + -2.87E-09*(T**3)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_N2
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_N2
            }
        },
        {
            "symbol": "CH4",
            "MW": MW_CH4,
            "Pc": Pc_CH4,
            "Tc": Tc_CH4,
            "w": w_CH4,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "19.875 + 5.021E-02*T + 1.268E-05*(T**2) + -11.004E-09*(T**3)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_CH4
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_CH4
            }
        },
        {
            "symbol": "C2H4",
            "MW": MW_C2H4,
            "Pc": Pc_C2H4,
            "Tc": Tc_C2H4,
            "w": w_C2H4,
            "Cp": {
                "unit": "kJ/kmol.K",
                "expr": "3.950 + 15.628E-02*T + -8.339E-05*(T**2) + 17.657E-09*(T**3)"
            },
            "dHf25": {
                "unit": "kJ/mol",
                "val": dHf25_C2H4
            },
            "dGf25": {
                "unit": "kJ/mol",
                "val": dGf25_C2H4
            }
        },
    ]
}

# database
componentData = componentDataStore['payload']


def retriveData(mode):
    pass


# component symbol
componentSymbolList = tuple([
    item['symbol'] for item in componentData])

# heat capacity at constant pressure [kJ/kmol.K]
heatCapacityAtConstatPresureList = tuple([
    {"symbol": item['symbol'], "Cp": item['Cp']['expr'], "unit": item['Cp']['unit']} for item in componentData])

# heat of formation [kJ/mol]
standardHeatOfFormationList = tuple([
    {"symbol": item['symbol'], "dHf25": item['dHf25']['val'], "unit": item['dHf25']['unit']} for item in componentData])
# print(standardHeatOfFormationList)

# standard Gibbs free energy of formation [kJ/mol]
standardGibbsFreeEnergyOfFormationList = tuple([
    {"symbol": item['symbol'], "dGf25": item['dGf25']['val'], "unit": item['dGf25']['unit']} for item in componentData])
# print(standardGibbsFreeEnergyOfFormationList)
