# STORE - STATE MANAGEMENT
# ------------------------

# import packages/modules
import json

#


class storeData:
    #! init
    def __init__(self):
        # * set
        self.data = self.initData()

    #! init json data
    def initData(self):
        # var
        payload = {}
        # try/except
        try:
            # database file
            appPath = "data\component.json"
            with open(appPath) as f:
                data = json.load(f)
                payload = data['payload'].copy()
                return payload
        except NameError:
            print(NameError)

    # end fun

    #! get data
    def getData(self):
        return self.data
