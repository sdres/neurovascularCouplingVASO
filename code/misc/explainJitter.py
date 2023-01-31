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
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
jitters = [0, 0.785, 1.570, 2.355]
dataPoints = {}


for k in range(1,5):

    possible = np.linspace(0,4.5,4)
    included = possible[:k]
    fig, ax = plt.subplots(1,1,figsize=(7.5,5))

    # Create figure and axis
    for i, shift in enumerate(included):
        # Create figure and axis
        plt.plot(x + shift, response, '--', color = colors[i], linewidth = 2, label = jitters[i])

        if shift == 0:
            start = 0
        else:
            start = 1

        selection = []
        for j in range(start,6):
            val = 6 * j
            tmp = np.where(x==val-shift)
            selection.append(tmp[0][0])

        xticks = np.arange(0, (response.shape[0]/10), tickFactor)
        plt.plot(xticks[start*2::2], response[selection], 'o', color = colors[i])

        xticks = np.arange(0, (response.shape[0]/10)+9, tickFactor)
        xlabels = ['V', 'B'] * 7 + ['V']

        ax.set_xticks(xticks, xlabels)
        dataPoints[jitters[i]] = response[selection]

    ax.set_ylabel('Response [a.u.]', fontsize=24)
    ax.set_yticks([])
    ax.set_xlabel('Volume', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=18)

    plt.legend(title = 'Jitters [s]', fontsize=14, title_fontsize=18)
    plt.savefig(f'./results/explainJitter{k}.png', bbox_inches = "tight")

    plt.show()

cbvJitters = {0.0: 0.0, 0.785: 2.355, 1.570: 1.570, 2.355: 0.785}

# Plot datapoints only
# Create figure and axis
fig, ax = plt.subplots(1,1,figsize=(7.5,5))


tr = 0.785
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']

for i, shift in enumerate(np.arange(0,4*tr,tr)):

    ticks = np.arange(tr*i, dataPoints[jitters[i]].shape[0]*(tr*4),tr*4)
    # print(ticks)
    plt.plot(ticks, dataPoints[cbvJitters[jitters[i]]], 'o', color = colors[i])


ax.set_ylabel('Response [a.u.]', fontsize=24)
ax.set_xlabel('Time', fontsize=24)
ax.set_yticks([])
ax.set_xticks([])


plt.savefig(f'./results/explainJitterCombined.png', bbox_inches = "tight")

plt.show()
