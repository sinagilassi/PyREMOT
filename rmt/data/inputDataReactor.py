# INPUT DATA
# reactor
# catalyst
# membrane
# -------------------

# ! packed-bed reactor
# packed reactor diameter [m]
rea_D = 0.0381
# reactor diameter [m]
reaW_D = 0.0025
# reactor wall thickness [m]
reaWall_D = 0.01
# bed height [m]
rea_L = 0.25
# bed porosity - voidage of the fixed bed
bed_por = 0.39
# reactor wall thermal conductivity [J/K.m.s]
# iron = 79;
# steel = 50;
kwall = 50

# ! catalyst
# catalyst particle diameter [m]
cat_d = 0.002
# catalyst particle density [kg/m^3]
cat_rho = 1982
# catalyst porosity
cat_por = 0.45
# catalyst tortuosity
cat_tor = 2
# fraction of solids
rea_solid = 1 - bed_por
# catalyst bulk density - mass of catalyst per reactor volume [kg/m^3]
bulk_rho = cat_rho*rea_solid
# thermal conductivity of catalyst [J/K.m.s]
therCop = 0.22
# catalyst specific heat capacity [J/kg.K]
cat_Cp = 960

# ! membrane
# membrane tube diameter [m2]
mem_D = 0.0254
# membrane thickness [m]
mem_t = 0.0001
# membrane area [m^2/m^3 reactor]
mem_A = 100  # 250
# H2O permeance [kmol/(s*m^2*Pa)]
QH2O = 5e-10
# H2O/H2 selectivity
SelH2OH2 = 30  # 10
# H2 permeance [kmol/(s*m^2*Pa)]
QH2 = QH2O/SelH2OH2
# bed specific area [m2/m3 solid]
av = 352
# membrane thermal conductivity [J/K.m.s]
kmem = 1
