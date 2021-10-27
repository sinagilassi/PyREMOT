# APP SETTING
# ------------
# define all constants used in this app

modelTypes = {
    "M1": {
        "id": "M1",
        "des":
            "packed bed reactor, isothermal, plug-flow, no pressure drop"
    },
    "M2": {
        "id": "M2",
        "des":
            "packed bed reactor, non-isothermal, plug-flow, no pressure drop"
    },
    "M3": {
        "id": "M3",
        "des":
            "batch reactor"
    },
    "M4": {
        "id": "M4",
        "des":
            "plug-flow reactor"
    },
    "M5": {
        "id": "M5",
        "des":
            "plug-flow heterogenous reactor"
    },
    "M6": {
        "id": "M6",
        "des":
            "dynamic plug-flow homogenous reactor"
    },
    "M7": {
        "id": "M7",
        "des":
            "steady-state plug-flow homogenous reactor [concentration base]"
    },
    "M8": {
        "id": "M8",
        "des":
            "steady-state plug-flow homogenous reactor [concentration base]"
    },
    "M9": {
        "id": "M9",
        "des":
            "dynamic plug-flow homogenous reactor [concentration base]"
    },
    "M10": {
        "id": "M10",
        "des":
            "dynamic plug-flow heterogenous reactor [concentration base]",
        "numerical": "ocm"
    },
    "M11": {
        "id": "M11",
        "des":
            "dynamic plug-flow heterogenous reactor [concentration base]",
        "numerical": "fdm"
    },
    "M12": {
        "id": "M12",
        "des":
            "steady-state plug-flow heterogenous reactor [concentration base]",
        "numerical": "fdm"
    },
    "M13": {
        "id": "M13",
        "des":
            "dynamic heterogenous reactor [two time domain]",
        "numerical": "fdm"
    },
    "M14": {
        "id": "M14",
        "des":
            "steady-state heterogenous model",
        "numerical": "fdm"
    },
    "T1": {
        "id": "T1",
        "des":
            "dynamic model of catalyst diffusion-reaction",
        "numerical": ""
    },
    "T2": {
        "id": "T2",
        "des":
            "homogenous reactor model",
        "numerical": ""
    },
}

M1 = modelTypes['M1']['id']
M2 = modelTypes['M2']['id']
M3 = modelTypes['M3']['id']
M4 = modelTypes['M4']['id']
M5 = modelTypes['M5']['id']
M6 = modelTypes['M6']['id']
M7 = modelTypes['M7']['id']
M8 = modelTypes['M8']['id']
M9 = modelTypes['M9']['id']
