# REACTION RATE EXPRESSION
# -------------------------

class rmtReactionClass:
    #
    def __init__(self, funBody):
        self.funBody = funBody

    def reactionRateFunSet(self, **kwargs):
        """
        build reaction rate function
        **kwargs: 
            temperature [K]
            pressure [Pa]
            mole fraction
        """
        # try/except
        try:
            # temperature  [K]
            T = kwargs['T']
            # pressure [Pa]
            P = kwargs['P']
            # mole fraction
            y = kwargs['y']
            # funbody
            funBodySet = self.funBody
            return eval(funBodySet)
        except Exception as e:
            raise
