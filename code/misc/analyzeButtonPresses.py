"""Investigate attention task performance."""

import numpy as np
import glob
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Define ROOT dir
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# define subjects to work on
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
plt.style.use('dark_background')

subList = []
runList = []
targetDetectedList = []

nrTargets = []

for sub in subs:

    logFiles = sorted(glob.glob(f'code/stimulation/{sub}/ses-*/{sub}_ses-*_run-*_neurovascularCoupling.log'))

    for log in logFiles:
        # print(f'Processing {log}')
        logFile = pd.read_csv(log, usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break

        # define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        logFile = pd.read_csv(log, sep='\t', skiprows=firstVolRow, names=ColNames)

        # initiate lists
        targetTimes = []
        buttonPresses = []
        targetDetected = []
        targetSwitch = False

        # loop over lines and fine stimulation start and stop times
        for index, row in logFile.iterrows():
            if not logFile['event'][index] != logFile['event'][index]:

                if (re.search('Target presented', logFile['event'][index])) and targetSwitch:
                    targetDetected.append(-1)

                if re.search('Target presented', logFile['event'][index]):
                    targetTimes.append(logFile['startTime'][index])
                    targetSwitch = True

                if re.search('Target detected', logFile['event'][index]):
                    targetDetected.append(logFile['startTime'][index])
                    targetSwitch = False

                if re.search('Keypress: 1', logFile['event'][index]):
                    buttonPresses.append(logFile['startTime'][index])

        count = 0
        for i, target in enumerate(targetDetected[1:]):
            # print(f'Comparing {target} with {targetDetected[i]}')
            if target == targetDetected[i]:
                count = count + 1
        if count >= 1:
            print(targetDetected)
            print(log)
            print(f'detected {count+1} missed targets in a row')

        # Count missed targets
        detectedRatio = (len(targetTimes) - targetDetected.count(-1))/len(targetTimes)

        nrTargets.append(len(targetTimes))
        subList.append(sub)
        runList.append(log.split('/')[-1][:20])
        targetDetectedList.append(detectedRatio)

np.min(nrTargets)
np.max(nrTargets)
np.mean(nrTargets)


data = pd.DataFrame({'subject': subList, 'run': runList, 'ratio': targetDetectedList})

subs = []
ratios = []
counts = []
for sub in data['subject'].unique():
    tmp = data.loc[data['subject'] == sub]
    for ratio in tmp['ratio'].unique():
        tmpData = tmp.loc[tmp['ratio'] == ratio]
        ratios.append(np.round(ratio, 2))
        counts.append(len(tmpData))
        subs.append(sub)

data2 = pd.DataFrame({'subject': subs, 'ratio': ratios, 'count': counts})


# Ignore participants
ratioList = []
countList = []

for ratio in data2['ratio'].unique():
    tmp = data2.loc[data2['ratio'] == ratio]['count'].to_numpy()
    val = np.sum(tmp)
    ratioList.append(ratio)
    countList.append(val)

newDat = pd.DataFrame({'ratio': ratioList, 'count': countList})


fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))
splot = sns.barplot(data=newDat, x='ratio', y='count', linewidth=1, edgecolor="1", color='tab:red')
# splot = sns.barplot(data=data2, x='ratio', y='count', linewidth=1, edgecolor="1", color='tab:red')
# splot = sns.barplot(data=data2, x='ratio', y='count', linewidth=1, edgecolor="1", hue='subject')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points',
                   fontsize=14)

plt.ylim(0, 90)
ax1.set_ylabel(r'# runs', fontsize=24)
ax1.set_xlabel(r'Ratio detected', fontsize=24)
ax1.yaxis.set_tick_params(labelsize=18)
ax1.xaxis.set_tick_params(labelsize=18)
fig.tight_layout()
plt.savefig(f'./results/targetsDetectedVsUndetected.png', bbox_inches="tight")

plt.show()
