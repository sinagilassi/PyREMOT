# BUILD FINITE ELEMENT MATRIX
# FOR CATALYST PARTICLES
# ----------------------------


# import package/modules
import numpy as np


class FiElCatParticleClass:
    def __init__(self, NuEl, NuToCoPo, hi, Xc, N, Q, A, B, odeNo):
        '''
        args:
            NuEl: number of elements
            NuToCoPo: number of total collocation points
            hi: element size list
            Xc: list of collocation points
            N: number of collocation points
            Q: OC Q matrix
            A: OC A matrix
            B: OC B matrix 
            odeNo: number of ode equations
        '''
        self.NuEl = NuEl
        self.NuToCoPo = NuToCoPo
        self.hi = hi
        self.Xc = Xc
        self.N = N
        self.Q = Q
        self.A = A
        self.B = B
        self.odeNo = odeNo

    # NOTE
    # BC1
    def fRbc1(self, i, j, Aij, Bij, N, h, const1, const2, const3):
        '''
        inlet boundary condition
        args:
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
        if i == 0:
            # bc1
            F = (1/h)*Aij
            f = 6  # careful
        elif i > 0 and i < N-1:
            # interior points
            F = (1/6)*(1/(h**2))*Bij - (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        elif i == N-1:
            # continuity term
            F = (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        else:
            raise

        # res
        res = np.array([F, f])
        # return
        return res

    # NOTE
    # BC2
    def fRbc2(self, i, j, Aij, Bij, N, h, const1, const2, const3):
        '''
        outlet boundary condition
        args:
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
        if i == 0:
            # bc1
            F = (1/h)*Aij
            f = 0
        elif i > 0 and i < N-1:
            # interior points
            F = (1/6)*(1/(h**2))*Bij - (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        elif i == N-1:
            # continuity term
            F = (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        else:
            raise

        # res
        res = np.array([F, f])
        # return
        return res

    # NOTE
    # interrior elements
    def fR(self, i, j, Aij, Bij, N, h, const1, const2, const3):
        '''
        interrior oc points
        args:
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
        if i == 0:
            # bc1
            F = (1/h)*Aij
            f = 0
        elif i > 0 and i < N-1:
            # interior points
            F = (1/6)*(1/(h**2))*Bij - (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        elif i == N-1:
            # continuity term
            F = (1/h)*Aij
            f = 0  # *** maybe contains nonlinear term ***
        else:
            raise

        # res
        res = np.array([F, f])
        # return
        return res

    # NOTE

    def fillElMat(self, k, h, const1, const2, const3):
        '''
        fill element matrix
        args:
            k: element number
            h: element length h[i]
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
            # R matrix
            RMatShape = (self.N, self.N)
            R = np.zeros(RMatShape)
            # f matrix
            fMatShape = (self.N, 1)
            f = np.zeros(fMatShape)
            # fill
            # A[i,j]
            for i in range(self.N):
                for j in range(self.N):
                    # check bc
                    if k == 0:
                        # first element
                        _loopVar = self.fRbc1(
                            i, j, self.A[i, j], self.B[i, j], self.N, h, const1, const2, const3)

                    elif k > 0 and k < self.NuEl-1:
                        # interrior elements
                        _loopVar = self.fR(
                            i, j, self.A[i, j], self.B[i, j], self.N, h, const1, const2, const3)
                    elif k == self.NuEl-1:
                        # last element
                        _loopVar = self.fRbc2(
                            i, j, self.A[i, j], self.B[i, j], self.N, h, const1, const2, const3)
                    else:
                        raise

                    # fill R
                    R[i, j] = _loopVar[0]
                # fill f
                f[i, 0] = _loopVar[1]

            # res
            res = {
                "R": R,
                "f": f
            }

            # return
            return res
        except Exception as e:
            raise

    def ResMatContinuity(self, CoMat):
        '''
        set matrix continuity
        args: 
            CoMat: coefficient matrix
        '''
        try:
            # continuity modifier matrix
            CoMoMat = np.ones((self.N, self.N))
            CoMoMat[0, :] = -1
            # edge matrix
            EdMat = np.zeros((self.N, self.N))
            # element matrix
            # ElMatsShape = (self.NuEl, self.NuToCoPo, self.NuToCoPo)
            # ElMats = np.zeros(ElMatsShape)
            ElMatShape = (self.NuToCoPo, self.NuToCoPo)
            ElMat = np.zeros(ElMatShape)
            #
            for k in range(self.NuEl):
                # set position
                _i0 = 0 if k == 0 else (k-1)*self.N + self.N - 1*k
                _i1 = _i0 + self.N
                _j0 = _i0
                _j1 = _i1
                _CoMatLoop = CoMat[k, :, :]

                # check
                if k == 0:
                    ElMat[0:self.N, 0:self.N] = _CoMatLoop
                else:
                    _loopVar0 = np.multiply(_CoMatLoop, CoMoMat)
                    _loopVar1 = EdMat + _loopVar0
                    ElMat[_i0:_i1, _j0:_j1] = _loopVar1

                # edge element set
                EdMat[0, 0] = ElMat[_i1-1, _j1-1]

            # res
            return ElMat

        except Exception as e:
            raise

    def fMatContinuity(self, fMat):
        '''
        set matrix continuity
        args: 
            fMat: nonlinear term matrix
        '''
        try:
            # continuity modifier matrix
            CoMoMat = np.ones((self.N, 1))
            CoMoMat[0, 0] = -1
            # edge matrix
            EdMat = np.zeros((self.N, 1))
            # element matrix
            ElMatShape = (self.NuToCoPo, 1)
            ElMat = np.zeros(ElMatShape)
            #
            for k in range(self.NuEl):
                # set position
                _i0 = 0 if k == 0 else (k-1)*self.N + self.N - 1*k
                _i1 = _i0 + self.N
                _j0 = _i0
                _j1 = _i1
                _MatLoop = fMat[k, :, :]

                # check
                if k == 0:
                    ElMat[0:self.N, :] = _MatLoop
                else:
                    _loopVar0 = _MatLoop
                    ElMat[_i0:_i1, :] = _loopVar0

            # res
            return ElMat

        except Exception as e:
            raise

    def initMatrix(self, const1, const2, const3):
        '''
        initialize matrix for each element
        '''
        try:
            # Ri matrix
            RiMatShape = (self.NuEl, self.N, self.N)
            Ri = np.zeros(RiMatShape)
            # fi matrix
            fiMatShape = (self.NuEl, self.N, 1)
            fi = np.zeros(fiMatShape)
            # fill matrices
            for k in range(self.NuEl):
                # set h
                h = self.hi[k]
                _loopVar = self.fillElMat(k, h, const1, const2, const3)
                # fill
                Ri[k, :, :] = _loopVar['R']
                fi[k, :, :] = _loopVar['f']

            # excert continuity
            # residual matrix
            _res0 = self.ResMatContinuity(Ri)
            # nonlinear/linear term
            _res1 = self.fMatContinuity(fi)

            # return
            res = {
                "Ri": _res0,
                "fi": _res1
            }
            # return
            return res
        except Exception as e:
            raise

    def buildMatrix(self, yj, const1=(), const2=(), const3=(), mode="default"):
        '''
        build matrix
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
            # run
            res = self.initMatrix(const1, const2, const3)
            # return
            return res
        except Exception as e:
            raise
