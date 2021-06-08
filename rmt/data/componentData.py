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

# critical pressure [bar]
Pc_CO2 = 73.74
Pc_H2 = 12.97
Pc_CH3OH = 80.97
Pc_H2O = 220.64
Pc_CO = 34.94
Pc_DME = 53
Pc_N2 = 33.98

# acentric factor
w_CO2 = 0.239
w_H2 = -0.216
w_CH3OH = 0.556
w_H2O = 0.344
w_CO = 0.066
w_DME = 0.200
w_N2 = 0.039

# component database
componentDataStore = {
    "payload": [
        {
            "symbol": "CO2",
            "MW": MW_CO2,
            "Pc": Pc_CO2,
            "Tc": Tc_CO2,
            "w": w_CO2
        },
        {
            "symbol": "H2",
            "MW": MW_H2,
            "Pc": Pc_H2,
            "Tc": Tc_H2,
            "w": w_H2
        },
        {
            "symbol": "CH3OH",
            "MW": MW_CH3OH,
            "Pc": Pc_CH3OH,
            "Tc": Tc_CH3OH,
            "w": w_CH3OH
        },
        {
            "symbol": "H2O",
            "MW": MW_H2O,
            "Pc": Pc_H2O,
            "Tc": Tc_H2O,
            "w": w_H2O
        },
        {
            "symbol": "CO",
            "MW": MW_CO,
            "Pc": Pc_CO,
            "Tc": Tc_CO,
            "w": w_CO
        },
        {
            "symbol": "DME",
            "MW": MW_DME,
            "Pc": Pc_DME,
            "Tc": Tc_DME,
            "w": w_DME
        },
        {
            "symbol": "N2",
            "MW": MW_N2,
            "Pc": Pc_N2,
            "Tc": Tc_N2,
            "w": w_N2
        }
    ]
}
