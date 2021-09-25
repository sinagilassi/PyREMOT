# FINITE DIFFERENCE METHOD
# -------------------------

# import module/packages
import numpy as np

# number of elements
NoEl_FiDi = 10


def FiDiBuildCMatrix(compNo, DoLe, yi, **args):
    '''
    build concentration residual matrix [R]
    args:
        compNo: component no
        DoLe: domain length [m]
        **args: 
            DiCoi: diffusivity coefficient of components [m^2/s]
            MaTrCoi: mass transfer coefficient [m/s]
            Ri: formation rate of component [kmol/m^3.s] | [mol/m^3.s]
            SpCoiBulk: species concentration of component in the bulk phase [kmol/m^3] | [mol/m^3]
    '''
    # try/except
    try:
        # number of finite differene points
        Ni = NoEl_FiDi + 1
        # number of total nodes
        N = Ni*compNo
        # dr size [m]
        dr = DoLe/NoEl_FiDi
        # formula
        rp = DoLe

        # NOTE
        ### matrix structure ##
        # residual matrix
        AMatShape = (compNo, Ni)
        A = np.zeros(AMatShape)

        # FIXME
        DiCoi = args['DiCoi']
        MaTrCoi = args['MaTrCoi']
        Ri = args['Ri']
        SpCoiBulk = args['SpCoiBulk']

        # concentration
        for c in range(compNo):
            # component yj
            SpCoi = yi[i, :]

            for i in range(Ni):
                # dr distance
                ri = i*dr
                # constant
                const1 = (DiCoi[i]/dr**2)
                const2 = 2*DiCoi[i]/(ri*2*dr)

                # reaction term
                _Ri = Ri[i]*(rp**2)

                if i == 0:
                    A[c][i] = 3*const1(2*SpCoi[i+1] - 2*SpCoi[i]) + _Ri

                elif i > 0 and i < Ni-1:
                    A[c][i] = const1*(SpCoi[i-1] - 2*SpCoi[i] + SpCoi[i+1]) + \
                        const2*(SpCoi[i+1] - SpCoi[i-1]) + _Ri
                elif i == Ni-1:
                    # const
                    alpha = MaTrCoi[i]/DiCoi[i]
                    # ghost point y[N+1]
                    yN__1 = (2*dr)*alpha*(SpCoi[i] - SpCoiBulk[i]) + SpCoi[i-1]
                    A[c][i] = const1*(SpCoi[i-1] - 2*SpCoi[i] + yN__1) + \
                        const2*(yN__1 - SpCoi[i-1]) + _Ri

        # res
        return A
    except Exception as e:
        raise


def FiDiBuildTMatrix(compNo, DoLe, yi, **args):
    '''
    build temperature residual matrix [R]
    args:
        compNo: component no
        DoLe: domain length [m]
        **args: 
            CaThCo: thermal conductivity of catalyst [kJ/s.m.K] | [J/s.m.K]
            HeTrCo: heat transfer coefficient [kJ/m^2.s.K] | [J/m^2.s.K] 
            Ri: formation rate of component [kmol/m^3.s] | [mol/m^3.s]
            dHi: enthalpy of reactions [kJ/kmol] | [J/kmol]
            TBulk: temperature of component in the bulk phase [K]
    '''
    # try/except
    try:
        # number of finite differene points
        Ni = NoEl_FiDi + 1
        # number of total nodes
        N = Ni*compNo
        # dr size [m]
        dr = DoLe/NoEl_FiDi
        # formula
        rp = DoLe

        # NOTE
        ### matrix structure ##
        # residual matrix
        AMatShape = (1, Ni)
        A = np.zeros(AMatShape)

        # FIXME
        CaThCo = args['CaThCo']
        HeTrCo = args['HeTrCo']
        Ri = args['Ri']
        dHi = args['dHi']
        TBulk = args['TBulk']

        # temperature
        # element of yj
        Ti = yi[0, :]

        # constant
        const1 = (CaThCo/dr**2)

        for i in range(Ni):
            # dr distance
            ri = i*dr
            const2 = 2*CaThCo/(ri*2*dr)

            # reaction term
            _Ri = Ri[i]
            _dHi = dHi[i]
            _dHRi = np.dot(_Ri, _dHi)*(rp**2)

            if i == 0:
                A[0][i] = 3*const1(2*Ti[i+1] - 2*Ti[i]) + _dHRi

            elif i > 0 and i < Ni-1:
                A[0][i] = const1*(Ti[i-1] - 2*Ti[i] + Ti[i+1]) + \
                    const2*(Ti[i+1] - Ti[i-1]) + _dHRi
            elif i == Ni-1:
                # const
                alpha = -1*HeTrCo/CaThCo
                # ghost point y[N+1]
                yN__1 = (2*dr)*alpha*(Ti[i] - TBulk[i]) + Ti[i-1]
                A[0][i] = const1*(Ti[i-1] - 2*Ti[i] + yN__1) + \
                    const2*(yN__1 - Ti[i-1]) + _dHRi

        # res
        return A
    except Exception as e:
        raise


def FiDiSetRMatrixBC1(i, j, dr, rp, yj, **argsVal):
    '''
    BC1 
    args:
        i: main index
        j: varied index
        dr: differential length [m]
        rp: particle radius [m]
        yj: variable 
        **argsVal
            concentration: 
                diffusivity coefficient [m^2/s]
                Ri: formation rate [mol/m^3.s] | [kmol/m^3.s]
    '''
    # concentration
    # for c in range(compNo):
    #     for i in range(Ni):

    #         # bc1
    #         if i == 0:
    #             for j in range(Ni):
    #                 A[c][i][j] = 1
    #         elif i > 0 and i < Ni-2:
    #             A[c][i][j] = 1
    #         elif i == Ni-1:
    #             A[c][i][j] = 1

    # diffusivity coefficient [m^2/s]
    DiCoi = argsVal['DiCoi']
    # formation rate [mol/m^3.s] | [kmol/m^3.s]
    Ri = argsVal['Ri']
    # dr distance
    ri = i*dr
    # constant
    const1 = (DiCoi/dr**2)
    const2 = 2*DiCoi/(ri*2*dr)

    if j == i-1:
        a = const1*yj[i-1] - const2*yj[i-1]
    elif j == i:
        a = const1*(-2)*yj[i]
    elif j == i+1:
        a = const1*yj[i+1] + const2*yj[i+1]
    else:
        a = 0
