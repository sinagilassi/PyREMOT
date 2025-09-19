# Python Reactor Modeling Tools

![PyPI - License](https://img.shields.io/pypi/l/PyREMOT) 
[![PyPI Downloads](https://static.pepy.tech/badge/pyremot/month)](https://pepy.tech/projects/pyremot)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a1TUPt_9H-l77yfuBVv1CXoA3uI4j1Rt?usp=sharing) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UN5nhSigfIZ95gYHU5hThCagL2iXcpdk?usp=sharing)

Python Reactor Modeling Tools (PyREMOT) is an open-source package which can be used for process simulation, optimization, and parameter estimation. The current version consists of homogeneous models for steady-state and dynamic conditions.

You can visit [dashboard](https://pyremot.herokuapp.com/) to build model input and load examples!

You can also run your modeling on Google Colaboratory. 

1- [Steady-state pseudo-homogeneous model](https://colab.research.google.com/drive/1a1TUPt_9H-l77yfuBVv1CXoA3uI4j1Rt?usp=sharing)

2- [Dynamic pseudo-homogeneous model](https://colab.research.google.com/drive/1UN5nhSigfIZ95gYHU5hThCagL2iXcpdk?usp=sharing)

## Getting started

You can install this package

```bash
  pip install PyREMOT
```

## Documentation

The main method is called as:

```python
    from PyREMOT import rmtExe

    # model inputs
    # using dashboard to build model inputs
    modelInput = {...}

    # run
    res = rmtExe(modelInput)
```

Check component list available in the current version:

```python
    from PyREMOT import rmtCom

    # display component list
    res = rmtCom()
    print(res)

```

PyREMOT UI dashboard contains some panels as:

1- MODEL SELECTION

2- COMPONENTS

    H2;CO2;H2O;CO;CH3OH;DME

this code is automatically converted to python as:

```python
  compList = ["H2","CO2","H2O","CO","CH3OH","DME"]
```

3- REACTIONS

define reactions as:

**_ Add ; in the end of each line _**

    CO2 + 3H2 <=> CH3OH + H2O;
    CO + H2O <=> H2 + CO2;
    2CH3OH <=> DME + H2O

then:

```python
    reactionSet = {
    "R1":"CO2+3H2<=>CH3OH+H2O",
    "R2":"CO+H2O<=>H2+CO2",
    "R3":"2CH3OH<=>DME+H2O"
    }
```

In order to define reaction rate expressions, there are two code sections as

**_ Add ; in the end of each line _**

a) define parameters:

    "CaBeDe" : CaBeDe;
    "RT": x['R_CONST']*x['T'];
    "K1": 35.45*math.exp(-1.7069e4/x['RT']);
    "K2": 7.3976*math.exp(-2.0436e4/x['RT']);
    "K3": 8.2894e4*math.exp(-5.2940e4/x['RT']);
    "KH2": 0.249*math.exp(3.4394e4/x['RT']);
    "KCO2": 1.02e-7*math.exp(6.74e4/x['RT']);
    "KCO": 7.99e-7*math.exp(5.81e4/x['RT']);
    "Ln_KP1": 4213/x['T'] - 5.752 * math.log(x['T']) - 1.707e-3*x['T'] + 2.682e-6 * (math.pow(x['T'], 2)) - 7.232e-10*(math.pow(x['T'], 3)) + 17.6;
    "KP1": math.exp(x['Ln_KP1']);
    "log_KP2": 2167/x['T'] - 0.5194 * math.log10(x['T']) + 1.037e-3*x['T'] - 2.331e-7 * (math.pow(x['T'], 2)) - 1.2777;
    "KP2": math.pow(10, x['log_KP2']);
    "Ln*KP3": 4019/x['T'] + 3.707 * math.log(x['T']) - 2.783e-3*x['T'] + 3.8e-7 * (math.pow(x['T'], 2)) - 6.56e-4/(math.pow(x['T'], 3)) - 26.64;
    "KP3": math.exp(x['Ln_KP3']);
    "yi*H2": x['MoFri'][0];
    "yi_CO2": x['MoFri'][1];
    "yi_H2O": x['MoFri'][2];
    "yi_CO": x['MoFri'][3];
    "yi_CH3OH": x['MoFri'][4];
    "yi_DME": x['MoFri'][5];
    "PH2": x['P']*(x['yi_H2'])_1e-5;
    "PCO2": x['P']_(x['yi_CO2'])_1e-5;
    "PH2O": x['P']_(x['yi_H2O'])_1e-5;
    "PCO": x['P']_(x['yi_CO'])_1e-5;
    "PCH3OH": x['P']_(x['yi_CH3OH'])_1e-5;
    "PCH3OCH3": x['P']_(x['yi_DME'])*1e-5;
    "ra1": x['PCO2']*x['PH2'];
    "ra2": 1 + (x['KCO2']*x['PCO2']) + (x['KCO']*x['PCO']) + math.sqrt(x['KH2']_x['PH2']);
    "ra3": (1/x['KP1'])_((x['PH2O']_x['PCH3OH'])/(x['PCO2']_(math.pow(x['PH2'], 3))));
    "ra4": x['PH2O'] - (1/x['KP2'])*((x['PCO2']*x['PH2'])/x['PCO']);
    "ra5": (math.pow(x['PCH3OH'], 2)/x['PH2O'])-(x['PCH3OCH3']/x['KP3'])

then converted:

```python
   varis0 = {
   "CaBeDe" : CaBeDe,
   "RT": lambda x: x['R_CONST']*x['T'],
   "K1": lambda x: 35.45*math.exp(-1.7069e4/x['RT']),
   "K2": lambda x: 7.3976*math.exp(-2.0436e4/x['RT']),
   "K3": lambda x: 8.2894e4*math.exp(-5.2940e4/x['RT']),
   "KH2": lambda x: 0.249*math.exp(3.4394e4/x['RT']),
   "KCO2": lambda x: 1.02e-7*math.exp(6.74e4/x['RT']),
   "KCO": lambda x: 7.99e-7*math.exp(5.81e4/x['RT']),
   "Ln_KP1": lambda x: 4213/x['T'] - 5.752 *     math.log(x['T']) - 1.707e-3*x['T'] + 2.682e-6 *     (math.pow(x['T'], 2)) - 7.232e-10*(math.pow(x['T'], 3)) + 17.6,
   "KP1": lambda x: math.exp(x['Ln_KP1']),
   "log_KP2": lambda x: 2167/x['T'] - 0.5194 *     math.log10(x['T']) + 1.037e-3*x['T'] - 2.331e-7 *     (math.pow(x['T'], 2)) - 1.2777,
   "KP2": lambda x: math.pow(10, x['log_KP2']),
       "Ln_KP3": lambda x:  4019/x['T'] + 3.707 *     math.log(x['T']) - 2.783e-3*x['T'] + 3.8e-7 *     (math.pow(x['T'], 2)) - 6.56e-4/(math.pow(x['T'], 3)) - 26.64,
   "KP3": lambda x:  math.exp(x['Ln_KP3']),
   "yi_H2": lambda x:  x['MoFri'][0],
   "yi_CO2": lambda x:  x['MoFri'][1],
   "yi_H2O": lambda x:  x['MoFri'][2],
   "yi_CO": lambda x:  x['MoFri'][3],
   "yi_CH3OH": lambda x:  x['MoFri'][4],
   "yi_DME": lambda x:  x['MoFri'][5],
   "PH2": lambda x:  x['P']*(x['yi_H2'])*1e-5,
   "PCO2": lambda x:  x['P']*(x['yi_CO2'])*1e-5,
   "PH2O": lambda x:  x['P']*(x['yi_H2O'])*1e-5,
   "PCO": lambda x: x['P']*(x['yi_CO'])*1e-5,
   "PCH3OH": lambda x:  x['P']*(x['yi_CH3OH'])*1e-5,
   "PCH3OCH3": lambda x:  x['P']*(x['yi_DME'])*1e-5,
   "ra1": lambda x:  x['PCO2']*x['PH2'],
   "ra2": lambda x:  1 + (x['KCO2']*x['PCO2']) + (x['KCO']*x['PCO']) + math.sqrt(x['KH2']*x['PH2']),
   "ra3": lambda x: (1/x['KP1'])*((x['PH2O']*x['PCH3OH'])/(x['PCO2']*(math.pow(x['PH2'], 3)))),
   "ra4": lambda x:  x['PH2O'] - (1/x['KP2'])*((x['PCO2']*x['PH2'])/x['PCO']),
   "ra5": lambda x: (math.pow(x['PCH3OH'], 2)/x['PH2O'])-(x['PCH3OCH3']/x['KP3'])
   }
```

b) define the final form of reaction rate expressions:

    "r1": 1000*x['K1']*(x['ra1']/(math.pow(x['ra2'], 3)))*(1-x['ra3'])*x['CaBeDe'];
    "r2": 1000*x['K2']*(1/x['ra2'])*x['ra4']*x['CaBeDe'];
    "r3": 1000*x['K3']*x['ra5']*x['CaBeDe']

then converted:

```python
   rates0 = {
   "r1": lambda x: 1000*x['K1']*(x['ra1']/(math.pow(x['ra2'], 3)))*(1-x['ra3'])*x['CaBeDe'],
   "r2": lambda x: 1000*x['K2']*(1/x['ra2'])*x['ra4']*x['CaBeDe'],
   "r3": lambda x: 1000*x['K3']*x['ra5']*x['CaBeDe']
   }
```

4- PROPERTIES

feed properties:

```python
   # species-concentration [mol/m^3]
   SpCoi = [574.8978, 287.4489, 1.15e-02, 287.4489, 1.15e-02, 1.15e-02]
   # flowrate @ P & T [m^3/s]
   VoFlRa = 0.000228
   # pressure [Pa]
   P = 5000000
   # temperature [K]
   T = 523
   # process-type [-]
   PrTy = "non-iso-thermal"
```

5- REACTOR

reactor and catalyst characteristics:

```python
   # reactor-length [m]
   ReLe = 1
   # reactor-inner-diameter [m]
   ReInDi = 0.0381
   # bed-void-fraction [-]
   BeVoFr = 0.39
   # catalyst bed density [kg/m^3]
   CaBeDe = 1171.2
   # particle-diameter [m]
   PaDi = 0.002
   # particle-density [kg/m^3]
   CaDe = 1920
   # particle-specific-heat-capacity  [J/kg.K]
   CaSpHeCa = 960
```

6- HEAT-EXCHANGER

```python
    # overall-heat-transfer-coefficient [J/m^2.s.K]
    U = 50
    # medium-temperature [K]
    Tm = 523
```

7- SOLVER

```python
    # ode-solver [-]
    ivp = "default"
    # display-result [-]
    diRe = "True"
```

After setting all modules, you can find 'model input' in python format in the summary panel. Then, copy the content of this file in your python framework and run it!

You can also find an example on PyREMOT dashboard, load it and then have a look at the summary panel.

## Run

As the downloaded python file contains modelInput variable, you can directly run the model as:

```python
    # model input
    modelInput = {...}
    # start modeling
    res = rmtExe(modelInput)
```

## Result Format

For steady-state cases, the modeling result is stored in an array named dataPack:

```python
    # res
    dataPack = []
    dataPack.append({
        "modelId": modelId,
        "processType": processType,
        "successStatus": successStatus,
        "computation-time": elapsed,
        "dataShape": dataShape,
        "labelList": labelList,
        "indexList": indexList,
        "dataTime": [],
        "dataXs": dataXs,
        "dataYCons1": dataYs_Concentration_DiLeVa,
        "dataYCons2": dataYs_Concentration_ReVa,
        "dataYTemp1": dataYs_Temperature_DiLeVa,
        "dataYTemp2": dataYs_Temperature_ReVa,
        "dataYs": dataYs_All
    })
```

And for dynamic cases,

```python
    # res
    resPack = {
        "computation-time": elapsed,
        "dataPack": dataPack
    }
```

Concentration results:

    dataYCons1: dimensionless concentration

    dataYCons2: concentration [mol/m^3.s]

Temperature results:

    dataYTemp1: dimensionless temperature

    dataYTemp2: Temperature [K]

All modeling results is also saved in dataYs.

## FAQ

For any question, you can contact me on [LinkedIn](https://www.linkedin.com/in/sina-gilassi/) or [Twitter](https://twitter.com/sinagilassi).

## Authors

- [@sinagilassi](https://www.github.com/sinagilassi)
