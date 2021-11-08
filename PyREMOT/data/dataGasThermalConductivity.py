# GAS THERMAL CONDUCTIVITY
# -------------------------


# ref: Perry's Chemical Engineers' Handbook,
# correlations for selected compounds at low pressures
# unit: W/m.K
GasTherConductivityData = [
    {
        "symbol": "CO2",
        "eqParams": [3.69, -0.3838, 964, 1860000],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 194.67,
            "max": 1500
        }
    },
    {
        "symbol": "H2",
        "eqParams": [0.002653, 0.7452, 12, 0],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 22,
            "max": 1600
        }
    },
    {
        "symbol": "CH3OH",
        "eqParams": [5.7992E-07, 1.7862, 0, 0],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 240,
            "max": 1000
        }
    },
    {
        "symbol": "H2O",
        "eqParams": [6.2041E-06, 1.3973, 0, 0],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 273.16,
            "max": 1073.15
        }
    },
    {
        "symbol": "CO",
        "eqParams": [0.00059882, 0.6863, 57.13, 501.92],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 70,
            "max": 1500
        }
    },
    {
        "symbol": "N2",
        "eqParams": [0.00033143, 0.7722, 16.323, 373.72],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 63.15,
            "max": 2000
        }
    },
    {
        "symbol": "CH4",
        "eqParams": [8.3983E-06, 1.4268, -49.654, 0],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 111.63,
            "max": 600
        }
    },
    {
        "symbol": "C2H4",
        "eqParams": [8.6806E-06, 1.4559, 299.72, -29, 403],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 170,
            "max": 590.92
        }
    },
    {
        "symbol": "C3H6",
        "eqParams": [0.0000449, 1.2018, 421, 0],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 225.45,
            "max": 1000
        }
    },
    {
        "symbol": "C3H8",
        "eqParams": [-1.12, 0.10972, -9834.6, -7535800],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 231.11,
            "max": 1000
        }
    },
    {
        "symbol": "C4H10",
        "eqParams": [0.051094, 0.45253, 5455.5, 1979800],
        "eqExpr": '',
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 272.65,
            "max": 1000
        }
    },
    {
        "symbol": "DME",
        "eqParams": [0.059975, 0.2667, 1018.6, 1098800],
        "eqExpr": "",
        "unit": "W/m.K",
        "range": {
            "unit": "K",
            "min": 248.31,
            "max": 1500
        }
    },
]

# viscosity equation list
TherConductivityList = tuple([{"symbol": item['symbol'], "eqParams": item['eqParams'],
                               "eqExpr": item['eqExpr'], "unit": item['unit']} for item in GasTherConductivityData])
