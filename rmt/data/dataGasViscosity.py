# GAS VISCOSITY DATA
# --------------------


# ref: chemical thermodynamics for process simulation
# Vapor viscosity correlations for selected compounds at low pressures
# unit: Pa.s
eq1GasViscosityData = [
    {
        "symbol": "CO2",
        "viscosity": [4.719875, 0.373279, 512.686300, -6119.961],
        "range": {
            "unit": "K",
            "min": 223,
            "max": 1473
        }
    },
    {
        "symbol": "H2",
        "viscosity": [0.169104, 0.692485, -7.634394, 467.120],
        "range": {
            "unit": "K",
            "min": 78,
            "max": 3000
        }
    },
    {
        "symbol": "CH3OH",
        "viscosity": [0.477915, 0.641076, 284.838034, -3230.713],
        "range": {
            "unit": "K",
            "min": 240,
            "max": 1000
        }
    },
    {
        "symbol": "H2O",
        "viscosity": [0.501246, 0.709247, 869.465599, -90063.891],
        "range": {
            "unit": "K",
            "min": 278,
            "max": 1173
        }
    },
    {
        "symbol": "CO",
        "viscosity": [0.734306, 0.588574, 52.318660, 1018.822],
        "range": {
            "unit": "K",
            "min": 68,
            "max": 1473
        }
    },
    {
        "symbol": "N2",
        "viscosity": [0.847662, 0.574033, 75.437536, 56.771],
        "range": {
            "unit": "K",
            "min": 73,
            "max": 1773
        }
    },
    {
        "symbol": "CH4",
        "viscosity": [1.119178, 0.493234, 214.627200, -3952.087],
        "range": {
            "unit": "K",
            "min": 93,
            "max": 1000
        }
    },
    {
        "symbol": "C2H4",
        "viscosity": [1.503552, 0.456140, 288.342422, 73.362],
        "range": {
            "unit": "K",
            "min": 170,
            "max": 1000
        }
    },
]

#
eq2GasViscosityData = [
    {
        "symbol": "DME",
        "viscosity": [2.68e-7, 0.3975, 534],
        "range": {
            "unit": "K",
            "min": 223,
            "max": 1473
        }
    },
]
