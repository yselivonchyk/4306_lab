import time
import os


def init(isFastRun):
    global plotLocation
    plotLocation =  './plots/'

    global currentPlotLocation
    subfolder = time.strftime("%Y.%m.%d_%H.%M.%S") + ('' if not isFastRun else '_fast')
    currentPlotLocation = './plots/' + subfolder + '/'

    if not os.path.exists(currentPlotLocation):
        os.makedirs(currentPlotLocation)