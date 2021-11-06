# GAS VISCOSITY DATA
# --------------------


# ref: chemical thermodynamics for process simulation
# Vapor viscosity correlations for selected compounds at low pressures
# unit: Pa.s
GasViscosityData = [
    {
        "symbol": "CO2",
        "eqParams": [4.719875, 0.373279, 512.686300, -6119.961],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 223,
            "max": 1473
        }
    },
    {
        "symbol": "H2",
        "eqParams": [0.169104, 0.692485, -7.634394, 467.120],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 78,
            "max": 3000
        }
    },
    {
        "symbol": "CH3OH",
        "eqParams": [0.477915, 0.641076, 284.838034, -3230.713],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 240,
            "max": 1000
        }
    },
    {
        "symbol": "H2O",
        "eqParams": [0.501246, 0.709247, 869.465599, -90063.891],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 278,
            "max": 1173
        }
    },
    {
        "symbol": "CO",
        "eqParams": [0.734306, 0.588574, 52.318660, 1018.822],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 68,
            "max": 1473
        }
    },
    {
        "symbol": "N2",
        "eqParams": [0.847662, 0.574033, 75.437536, 56.771],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 73,
            "max": 1773
        }
    },
    {
        "symbol": "CH4",
        "eqParams": [1.119178, 0.493234, 214.627200, -3952.087],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 93,
            "max": 1000
        }
    },
    {
        "symbol": "C2H4",
        "eqParams": [1.503552, 0.456140, 288.342422, 73.362],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 170,
            "max": 1000
        }
    },
    {
        "symbol": "C3H6",
        "eqParams": [0.876767, 0.520871, 293.618650, -182.857],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 88,
            "max": 1000
        }
    },
    {
        "symbol": "C3H8",
        "eqParams": [0.173966, 0.734798, 143.207060, -7147.859],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 85,
            "max": 1000
        }
    },
    {
        "symbol": "C4H10",
        "eqParams": [0.075828, 0.837082, 67618677, -2141.762],
        "eqExpr": '',
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 143,
            "max": 963
        }
    },
    {
        "symbol": "DME",
        "eqParams": [],
        "eqExpr": "2.68e-7*(T**0.3975)/(1+(534/T))",
        "unit": "Pa.s",
        "range": {
            "unit": "K",
            "min": 223,
            "max": 1473
        }
    },
]

# viscosity equation list
viscosityList = tuple([
    {"symbol": item['symbol'], "eqParams": item['eqParams'], "eqExpr": item['eqExpr'], "unit": item['unit']} for item in GasViscosityData
])
