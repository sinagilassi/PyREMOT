# GENERAL EQUATIONS
# ------------------


# import packages/modules
from core import constants as CONST
# add err class

#


class rmtUtilityClass:
    #
    def __init__(self) -> None:
        pass

    @staticmethod
    def mixtureMolecularWeight(MoFri, MWi):
        """
            calculate mixture molecular weight
        """
        # check
        if len(MoFri) != len(MWi):
            raise Exception("elements are not equal")

        MixMoWe = 1
        return MixMoWe

    @staticmethod
    def volumetricFlowrateSTP(VoFlRa, P, T):
        """
            calculate volumetric flowrate at STP conditions
            VoFlRa []
            P [Pa]
            T [K]
            VoFlRaSTP []
        """
        VoFlRaSTP = VoFlRa*(P/CONST.Pstp)*(CONST.Tstp/T)
        return VoFlRaSTP

    @staticmethod
    def VoFlRaSTPToMoFl(VoFlRaSTP):
        """
            convert volumetric flowrate [stp] to molar flowrate [ideal gas]
            VoFlRaSTP [m^3/s]
            MoFlRaIG [mol/s]
        """
        MoFlRaIG = (VoFlRaSTP/0.02241)
        return MoFlRaIG

    @staticmethod
    def reactorCrossSectionArea(BePo, ReDi):
        """
            calculate reactor cross section area 
            BePo: bed porosity [-]
            ReDi: reactor diameter [m]
            ReCrSeAr [m^2]
        """
        ReCrSeAr = BePo*(CONST.PI_CONST*(ReDi**2)/4)
        return ReCrSeAr
