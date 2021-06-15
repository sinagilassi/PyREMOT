# CONFIG APP
# -----------

# import packages/modules


# app config
appConfig = {
    "calculation": {
        "roundAccuracy": 3,
        "roundAccuracyMole": 4,
        "roundAccuracyConcentration": 4
    }
}

# round function accuracy
ROUND_FUN_ACCURACY = appConfig['calculation']['roundAccuracy']
# mole fraction accuracy
MOLE_FRACTION_ACCURACY = appConfig['calculation']['roundAccuracyMole']
# concentration
CONCENTRATION_ACCURACY = appConfig['calculation']['roundAccuracyConcentration']