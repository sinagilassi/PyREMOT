# BUILD ORTHOGONAL COLLOCATION MATRIX
# FOR CATALYST PARTICLES
# ------------------------------------


# import package/modules
import numpy as np


class OrCoCatParticle:
    def __init__(self, Xc, Q, A, B, odeNo, N):
        self.Xc = Xc
        self.Q = Q
        self.A = A
        self.B = B
        self.odeNo = odeNo
        self.N = N

    def CalCatGas(self, A, yj, Cb, Tb, beta, betaT):
        '''
        calculate concentration and temperature at the catalyst surface
            args:
                Cb: list of species concentration in the gas phase [kmol/m^3]
                Tb: temperature in the gas phase [K]
        '''
        # try/except
        try:
            # residual matrix shape
            _shape = (self.odeNo, 1)
            # define R matrix
            Y_BC2 = np.zeros(_shape)

            # concentration
            for k in range(self.odeNo - 1):
                # constant y[0 to N]*A[N+1,r]
                _Ay = np.dot(A[-1, :-1], yj[:-1])
                _AySum = np.sum(_Ay)
                _alpha = -1*(_AySum + beta[k]*Cb[k])
                # y[N+1] constant
                _AyBC2 = A[self.N, self.N] - beta[k]

                for i in range(self.N):
                    for j in range(self.N):
                        pass

        except Exception as e:
            raise

    # R (only linear coefficient)
    # first point is not BC1
    # last point is BC2

    def fR(self, i, j, Aij, Bij, N, k, beta, betaT):
        ''' 
        concentration equations:
            for BC2, add beta*C[bulk] by ff function
            reaction term is needed (all points)
        temperature:
            for BC2, add betaT*T[bulk]
            enthalpy of reaction is required (all points)
        '''
        # interior points [0,1,...,N-1]
        if k < self.odeNo - 1:
            # concentration eq.
            # BC2: N
            if i < N - 1:
                F = Bij + (2/self.Xc[i])*Aij
            # last node (BC2)
            else:
                if j == N - 1:
                    F = Aij - beta[k]
                else:
                    F = Aij
        else:
            # temperature eq.
            # BC2: N
            if i < N - 1:
                F = Bij + (2/self.Xc[i])*Aij
            # last node (BC2)
            else:
                if j == N - 1:
                    F = Aij - betaT
                else:
                    F = Aij
        return F

    def buildLhsMatrix(self, beta, betaT):
        # Residual Formulation
        # ---------------------
        # linear parts (RHS)
        # non-linear parts (LHS) appers in non-linear function
        # R1
        # R1 = [A(1,1)-6 A(1,2) A(1,3) A(1,4)]
        # R4
        # R4 = [A(4,1) A(4,2) A(4,3) A(4,4)]
        # R2/R3
        # R2 = [1/6*B(2,1)-A(2,1) 1/6*B(2,2)-A(2,2) 1/6*B(2,3)-A(2,3) 1/6*B(2,4)-A(2,4)]
        # R3 = [1/6*B(3,1)-A(3,1) 1/6*B(3,2)-A(3,2) 1/6*B(3,3)-A(3,3) 1/6*B(3,4)-A(3,4)]
        # R = [R1; R2; R3; R4]

        try:
            # residual matrix shape
            residualMatrixShape = (self.odeNo*self.N, self.odeNo*self.N)
            # define R matrix
            R = np.zeros(residualMatrixShape)

            for k in range(self.odeNo):
                # new row and column
                ij = k*self.N
                for i in range(self.N):
                    for j in range(self.N):
                        R[ij+i][ij+j] = self.fR(i, j, self.A[i]
                                                [j], self.B[i][j], self.N, k,  beta, betaT)

        except Exception as e:
            raise

    def ff(self, i, A, B,  N, k, beta, betaT, Cb, Tb):
        ''' 
        N: number of dependent variables
        '''
        if k < self.odeNo - 1:
            # concentration eq.
            if i < N-1:
                F = 0
            else:
                F = beta[k]*Cb
        else:
            # temperature eq.
            if i < N-1:
                F = 0
            else:
                F = betaT*Tb
        return F

    def buildRhsMatrix(self, beta, betaT, Cb, Tb):
        '''
        LHS of equation consisting nonlinear/linear term such as reaction
        args:
            beta list: var term [Rp*h/Di]
            betaT: var term [-1*Rp*hfs/therCoMixEff]
        '''
        try:
            # f matrix - constant values matrix
            # rhs
            rhsMatrixShape = self.odeNo*self.N
            # define f matrix
            f = np.zeros(rhsMatrixShape)
            for k in range(self.odeNo):
                # new row and column
                ij = k*self.N
                for i in range(self.N):
                    f[i] = self.ff(i, self.A, self.B, self.N, k)
        except Exception as e:
            raise

    def buildOrCoMatrix(self, compNo, zNo, rNo, beta, betaT, Cb, Tb):
        '''
        build main matrix used for df/dt
        args: 
            compNo, number of components
            zNo: number of finite difference points
            rNo: number of orthogonal collocation points

        '''
        try:
            # R matrix
            RMatrix = self.buildLhsMatrix(beta, betaT)
            # f matrix
            fMatrix = self.buildRhsMatrix(beta, betaT, Cb, Tb)
            # df/dt val
            dfdtValMatrix = np.zeros((compNo, rNo, zNo))

            for c in range(compNo):
                for i in range(zNo):
                    for j in range(rNo):
                        pass

        except Exception as e:
            raise
