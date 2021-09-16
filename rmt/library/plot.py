# PLOT RESULTS
# -------------

# import packages/modules
from typing import List
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
        # check data type
        if isinstance(data, List):
            lineNo = range(len(data))
            lineXs = [item['x'] for item in data]
            lineYs = [item['y'] for item in data]
            lineLegend = [item['leg'] if 'leg' in item.keys()
                          else "line" for item in data]

            for i in lineNo:
                plt.plot(lineXs[i], lineYs[i], label=lineLegend[i])
        else:
            lineXs = data['x']
            lineYs = data['y']
            if 'leg' in data.keys():
                lineLegend = data['leg']
            else:
                lineLegend = "line"

            plt.plot(lineXs, lineYs, label=lineLegend)

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
        args:
            XYList: (x,y) points
            labelList: name of component
        outlet:
            dataList:
                "x": points
                "y": points
                "leg": legend
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

    @staticmethod
    def plots2DSub(dataList, xLabel, yLabel, title="", dataListPoint=[]):
        """
        dataList:
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
        # plot no
        plotNo = len(dataList)
        # total number of
        # set subplot
        figure, axis = plt.subplots(plotNo)
        # load data
        for i in range(plotNo):
            # check data type
            if isinstance(dataList[i], List):
                lineNo = range(len(dataList[i]))
                lineXs = [item['x'] for item in dataList[i]]
                lineYs = [item['y'] for item in dataList[i]]
                lineLegend = [item['leg'] if 'leg' in item.keys()
                              else "line" for item in dataList[i]]

                for j in lineNo:
                    axis[i].plot(lineXs[j], lineYs[j], label=lineLegend[j])
                    axis[i].legend()
            else:
                lineXs = dataList[i]['x']
                lineYs = dataList[i]['y']
                if 'leg' in dataList[i].keys():
                    lineLegend = dataList[i]['leg']
                else:
                    lineLegend = "line"

                axis[i].plot(lineXs, lineYs, label=lineLegend)
                axis[i].legend()

        # points
        # dataListPoint
        if len(dataListPoint) > 0:
            # plot no
            plotNoPoint = len(dataListPoint)
            # set color
            # colors = np.random.rand(plotNoPoint)
            for i in range(plotNoPoint):
                # check data type
                if isinstance(dataListPoint[i], List):
                    lineNoPoint = range(len(dataListPoint[i]))
                    lineXsPoint = [item['x'] for item in dataListPoint[i]]
                    lineYsPoint = [item['y'] for item in dataListPoint[i]]
                    # lineLegendPoint = [item['leg'] if 'leg' in item.keys()
                    #                    else "line" for item in dataListPoint[i]]

                    for j in lineNoPoint:
                        axis[i].scatter(
                            lineXsPoint[j], lineYsPoint[j], alpha=0.5)
                else:
                    lineXsPoint = dataListPoint[i]['x']
                    lineYsPoint = dataListPoint[i]['y']
                    # if 'leg' in dataListPoint[i].keys():
                    #     lineLegend = dataList[i]['leg']
                    # else:
                    #     lineLegend = "line
                    axis[i].scatter(lineXsPoint, lineYsPoint, alpha=0.5)

        # title
        if len(title) > 0:
            plt.title(title)

        # labels
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        # # legend
        # plt.legend()

        # display
        plt.show()

    # @staticmethod
    # def plot2DSubRun(dataX, dataYs, labelList)
