# PLOT RESULTS
# -------------

# import packages/modules
import numpy as np
import matplotlib.pyplot as plt


class plotClass:
    """
        plot modeliing result
        plotData:
            X list
            Y list
            Z list
    """
    # init

    def __init__(self, plotData):
        self.plotData = plotData

    @staticmethod
    def plot2D(x, y, type="LINE"):
        """
            x: x point list
            y: y point list
            type: plot type 
        """

        # plot default
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plots2D(data, xLabel, yLabel, title=""):
        """
            data:
                data[i]:
                    xs[i]:
                        point list
                    yx[i]:
                        point list
                    linestyle (ls): solid (-), dotted (:), dashed (--), dashdor (-.)
                    color (c):
                    label: legend in the figure 
            xLabel: x axis name
            yLabel: y axis name
            title: plot title
        """
        lineNo = range(len(data))
        lineXs = [item['x'] for item in data]
        lineYs = [item['y'] for item in data]
        lineLegend = [item['leg'] if 'leg' in item.keys()
                      else "line" for item in data]

        for i in lineNo:
            plt.plot(lineXs[i], lineYs[i], label=lineLegend[i])

        # title
        if len(title) > 0:
            plt.title(title)

        # labels
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        # legend
        plt.legend()

        # display
        plt.show()

    @staticmethod
    def plots2DSetXYList(X, Ys):
        """
            build array of X,Y
        """
        lineList = [[X, item] for item in Ys]
        return lineList

    @staticmethod
    def plots2DSetDataList(XYList, labelList):
        """
            build array of dict X,Y,leg
        """
        dataList = []
        # line no
        lineNo = len(XYList)
        for i in range(lineNo):
            dataList.append({
                "x": XYList[i][0],
                "y": XYList[i][1],
                "leg": labelList[i]
            })
        # res
        return dataList
