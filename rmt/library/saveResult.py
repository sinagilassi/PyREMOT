# SAVE MODELING RESULT IN A FILE
# JSON,CSV,TEXT
# -------------------------------

# import modules/packages
import json
import csv

#


class saveResultClass:
    # init
    def __init__(self):
        pass

    #
    def saveListToText(data):
        """
            save data (list) to a text file
        """
        # check
        if not isinstance(data, list):
            # err
            print("data is not a list")
        else:
            textfile = open("saveFile.txt", "w")
            for element in data:
                textfile.write(str(element) + "\n")
            textfile.close()

    #
    def saveListToCSV(data, headerList):
        """
            save data (list) to a csv file
            data: List of list
            headerList: list of header name
        """
        # check
        if not isinstance(data, list):
            # err
            print("data is not a list")
        else:
            with open("saveFile.csv", "w", newline="") as f:
                write = csv.writer(f)
                write.writerow(headerList)
                write.writerows(data)
