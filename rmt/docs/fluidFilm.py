# Fluid–Film Coefficients
# -------------------------

# import packages/modules
from math import sqrt
import numpy as np


def main():
    pass


def calNuNoEq1(Pr, Re):
    """ 
    gas-solid Nusselt number
    args:
        Pr: Prandtl number
        Re: Reynolds number
    """
    # try/except
    try:
        return 2 + 1.1*(Pr**(1/3))*(Re**(0.6))
    except Exception as e:
        raise


def calShNoEq1(Sc, Re):
    """ 
    gas-solid Sherwood number
    args:
        Sc: Schmidt number
        Re: Reynolds number
    """
    # try/except
    try:
        return 2 + 1.1*(Sc**(1/3))*(Re**(0.6))
    except Exception as e:
        raise


def calReNoEq1(GaDe, SuVe, CaPaDi, GaVi):
    """ 
    calculate Reynolds number
    args:
        GaDe: gas density [kg/m^3]
        SuVe: superficial velocity [m/s]
        CaPaDi: catalyst particle diameter [m]
        GaVi: gas viscosity [Pa.s]
    """
    # try/except
    try:
        return SuVe*CaPaDi*GaDe/GaVi
    except Exception as e:
        raise


def calPrNoEq1(GaHeCaCoPr, GaVi, GaThCo, GaMoWe):
    """ 
    calculate Prandtl number
    args:
        GaHeCaCoPr: heat capacity at constant pressure [J/mol.K]
        GaThCo: gas thermal conductivity [J/m.s.K]
        GaVi: gas viscosity [Pa.s] | [kg/m.s]
        GaMoWe: gas molecular weight [g/mol]
    """
    # try/except
    try:
        # Cp conversion into [J/kg.K]
        GaHeCaCoPr1 = (GaHeCaCoPr/GaMoWe)*1000
        return GaHeCaCoPr1*GaVi/GaThCo
    except Exception as e:
        raise


def calScNoEq1(GaDe, GaVi, GaDiCoi):
    """ 
    calculate Schmidt number 
    args:
        GaDe: gas density [kg/m^3]
        GaVi: gas viscosity [Pa.s] | [kg/m.s]
        GaDiCoi: gas component diffusivity coefficient [m^2/s]
    """
    # try/except
    try:
        return (GaVi/GaDe)/GaDiCoi
    except Exception as e:
        raise


def calMassTransferCoefficientEq1(Sh, GaDiCoi, CaPaDi):
    """ 
    calculate mass transfer coefficient [m/s]
    args:
        Sh: Sherwood number
        GaDiCoi: gas component diffusivity coefficient [m^2/s]
        CaPaDi: catalyst particle diameter [m]
    """
    # try/except
    try:
        return (Sh/CaPaDi)*GaDiCoi
    except Exception as e:
        raise


def calHeatTransferCoefficientEq1(Nu, GaThCo, CaPaDi):
    """ 
    calculate heat transfer coefficient [J/m^2.s.K]
    args:
        Nu: Nusselt number
        GaThCo: GaThCo: gas thermal conductivity [J/m.s.K]
        CaPaDi: catalyst particle diameter [m]
    """
    # try/except
    try:
        return (Nu/CaPaDi)*GaThCo
    except Exception as e:
        raise


if __name__ == "__main__":
    main()
