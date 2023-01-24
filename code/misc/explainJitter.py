'''Make figures to explain jittering'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
import matplotlib

plt.style.use('dark_background')

# Create figure and axis
fig, ax = plt.subplots(1,1,figsize=(7.5,5))

# Define response function and x values
x = np.arange(0, 30, 0.1)
response = gamma.pdf(x, 8)

# Find maximum
maxIdx = np.argmax(response)
maxVal = np.max(response)

for i in range(40):
    response = np.insert(response,maxIdx+i,maxVal)

x = np.arange(0, response.shape[0]/10, 0.1)

plt.plot(x,response)

tickFactor = 3


xticks = np.arange(0, response.shape[0]/10, tickFactor)

xlabels = ['V', 'B'] * 6
ax.set_xticks(xticks, xlabels)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

jitters = [0, 0.785, 1.570, 2.355]

dataPoints = {}

for i, shift in enumerate(np.linspace(0,4.5,4)):

    plt.plot(x + shift, response, color = colors[i], linewidth = 2, label = jitters[i])

    if shift == 0:
        start = 0
    else:
        start = 1

    selection = []
    for j in range(start,6):
        val = 6 * j
        tmp = np.where(x==val-shift)
        selection.append(tmp[0][0])

    plt.plot(xticks[start*2::2], response[selection], 'o', color = colors[i])

    dataPoints[jitters[i]] = response[selection]

ax.set_ylabel('Response [a.u.]', fontsize=24)
ax.set_yticks([])
ax.set_xlabel('Volume', fontsize=24)
ax.xaxis.set_tick_params(labelsize=18)

plt.legend(title = 'Jitters [s]', fontsize=14, title_fontsize=18)

plt.show()
