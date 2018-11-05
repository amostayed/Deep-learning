import matplotlib
import numpy as np
#
def utility_plot(data, peaks):
    #
    peaks = np.array(peaks, dtype = int)
    t = np.arange(data.shape[1])
    channels = data.shape[0]
    fig = matplotlib.pyplot.figure(figsize = (10,10))
    #
    for i in np.arange(1, channels + 1):
        ax = matplotlib.pyplot.subplot(channels, 1, i)
        matplotlib.pyplot.plot(t, data[i - 1,:])
        matplotlib.pyplot.plot(peaks, data[i - 1, peaks], 'or')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
#