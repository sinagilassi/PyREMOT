# ERROR HANDLERES
# ----------------

class errHandlerClass(Exception):
    def __init__(self, *args: object):
        if args:
            self.message = args[0]
        else:
            self.message = None

    # def __str__(self):
    #     print('calling str')
    #     if self.message:
    #         return 'errClass, {0}'.format(self.message)
    #     else:
    #         return 'errClass has not raised'


# general
class errGeneralClass(errHandlerClass):
    def __init__(self, errCode, errMessage):
        self.errCode = errCode
        self.errMessage = errMessage

    def __str__(self):
        print('errGeneralClass')
        if self.errMessage:
            return f'errGeneralClass, {self.errMessage} {self.errCode}'
        else:
            return 'errGeneralClass has not raised'


# raise errGeneralClass(1, "typing error!")
