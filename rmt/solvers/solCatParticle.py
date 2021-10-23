# BUILD ORTHOGONAL COLLOCATION MATRIX
# FOR CATALYST PARTICLES
# ------------------------------------


# import package/modules
import numpy as np


class OrCoCatParticleClass:
    def __init__(self, Xc, N, Q, A, B, odeNo):
        '''
        args:
            Xc: list of collocation points
            N: number of collocation points
            Q: OC Q matrix
            A: OC A matrix
            B: OC B matrix 
            odeNo: number of ode equations
        '''
        self.Xc = Xc
        self.N = N
        self.Q = Q
        self.A = A
        self.B = B
        self.odeNo = odeNo

    def CalUpdateYnSolidGasInterface(self, yj, CTb, beta, fluxDir="lr"):
        '''
        calculate concentration and temperature at the catalyst surface
            args:
                yj: var matrix (species concentration or temperature at oc points)
                CTb: list of species concentration in the gas phase [kmol/m^3]
                beta: beta/betaT (for each component)
            output:
                yj_flip
        '''
        # try/except
        try:
            # yj
            # y[n], y[n-1], ..., y[0]
            # y solid-gas interface matrix shape
            _shape = (self.N, 1)
            # define C matrix
            yj_updated = np.zeros(_shape)
            # flip
            yj_flip = np.flip(yj).reshape(_shape)

            # yj shape
            yj_Shape = yj.reshape(_shape)

            if fluxDir == "rl":
                # concentration
                # constant y[0 to N]*A[N+1,r]
                _Ay_Selected = self.A[-1, :-1]
                _yj_Selected = yj_Shape[:-1, 0]
                _Ay0 = np.dot(_Ay_Selected, _yj_Selected)
                _Ay = _Ay_Selected*_yj_Selected
                _alpha = np.sum(_Ay) + beta*CTb
                # y[N+1] constant
                _gamma = beta - self.A[-1, -1]
                # updated concentration
                yn = _alpha/_gamma
                # update yj
                yj_Shape[-1][0] = yn
            elif fluxDir == "lr":
                # concentration
                # constant y[0 to N]*A[N+1,r]
                _Ay_Selected = self.A[-1, :-1]
                _yj_Selected = yj_Shape[:-1, 0]
                _Ay0 = np.dot(_Ay_Selected, _yj_Selected)
                _Ay = _Ay_Selected*_yj_Selected
                _Ay_Sum = np.sum(_Ay)
                _alpha = beta*CTb - _Ay_Sum
                # y[N+1] constant
                _gamma = beta + self.A[-1, -1]
                # updated concentration
                yn = _alpha/_gamma
                # update yj
                yj_Shape[-1][0] = yn

            # res
            return yj_Shape
        except Exception as e:
            raise

    # R (only linear coefficient)
    # first point is not BC1
    # last point is BC2

    def fR(self, i, j, Aij, Bij, N, contCT, constBeta=1):
        ''' 
        calculate element of R matrix
            args:
                contCT: 
                    concentration: effective diffusivity coefficient
                    temperature: effective thermal conductivity
                constBeta: 
                    concentration: dimensionless number
                    temperature: dimensionless number
        '''
        if i < N-1:
            # NOTE
            # interior points
            F = contCT*Bij + contCT*(2/self.Xc[i])*Aij
        elif i == N-1:
            # NOTE
            # BC2 point
            if j < N-1:
                F = Aij
            elif j == N-1:
                F = Aij + constBeta

        return F

    def buildLhsMatrix(self, contCT, constCT2):
        '''
        build Lhs (R) matrix
        args: 
            contCT: 
                concentration: effective diffusivity coefficient of component
                temperature: effective thermal conductivity 
            constCT2: 
                concentration:
                    Ci_c: bulk concentration 
                    _DiLeNu: dimensionless number
        '''
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
            residualMatrixShape = (self.N, self.N)
            # define R matrix
            R = np.zeros(residualMatrixShape)

            for i in range(self.N):
                for j in range(self.N):
                    R[i][j] = self.fR(i, j, self.A[i][j],
                                      self.B[i][j], self.N, contCT, constCT2[1])
            # res
            return R
        except Exception as e:
            raise

    def ff(self, i, N, contCT, constBeta=[0, 0]):
        ''' 
        calculate element of f matrix
        args:
            contCT: 
                concentration: reaction term
                temperature: overall enthalpy of reaction
            constBeta: 
                concentration: dimensionless number x Cb[i]
                temperature: dimensionless number x Tb
        '''
        if i < N-1:
            # NOTE
            # interrior points
            F = contCT
        elif i == N-1:
            # NOTE
            # BC2 point
            F = -1*constBeta[0]*constBeta[1]

        return F

    def buildRhsMatrix(self, contCT, constCT2):
        '''
        RHS of equation consisting nonlinear/linear term such as reaction
        args:
            contCT
                concentration: list of reaction term
                temperature: list of overall enthalpy of reaction
            constCT2: 
                concentration:
                    Ci_c: bulk concentration 
                    _DiLeNu: dimensionless number
        '''
        try:
            # f matrix - constant values matrix
            # rhs
            rhsMatrixShape = (self.N)
            # define f matrix
            f = np.zeros(rhsMatrixShape)

            for i in range(self.N):
                f[i] = self.ff(i, self.N, contCT[i], constCT2)

            # res
            return f
        except Exception as e:
            raise

    def buildOrCoMatrix(self, yj, const1, const2, const3=(), mode="default"):
        '''
        build main matrix used for df/dt
        args: 
            yj: var list at each OC point - shape: (rNo,1)
            const1:
                concentration: effective diffusivity coefficient
                temperature: effective thermal conductivity
            const2: 
                concentration: reaction term
                temperature: overall enthalpy of reaction
            const3: 
                concentration: bulk concentration & dimensionless number
                temperature: bulk temperature & dimensionless number
        '''
        try:
            # # yj
            # y[0], y[1], ..., y[n]
            # R matrix
            RMatrix = self.buildLhsMatrix(const1, const3)
            # f matrix
            fMatrix = self.buildRhsMatrix(const2, const3)

            # [R][Y]=[RY]
            RYMatrix = np.matmul(RMatrix, yj)

            # sum of R and F
            RYFMatrix = RYMatrix + fMatrix

            # should be flip C[n], C[n-1], ..., C[0]
            RYFMatrix_flip = np.flipud(
                RYFMatrix) if mode == "default" else RYFMatrix

            # res
            return RYFMatrix_flip

        except Exception as e:
            raise

# NOTE
# steady-state OC
