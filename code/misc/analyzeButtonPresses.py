'''

Investigate attention task performance.

'''

import numpy as np
import glob
import pandas as pd
import os
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

for sub in subs:
    # # get all runs of all sessions
    # runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-*_task-*_run-0*_*part-mag*_cbv.nii.gz'))
    #
    # for run in runs:
    #     # get basename of current run
    #     base = os.path.basename(run).rsplit('.', 2)[0][:-4]
    #     # see session in which it was acquired
    #
    #     for i in range(1,99):
    #         if f'ses-0{i}' in base:
    #             ses = f'ses-0{i}'

    logFiles = sorted(glob.glob(f'code/stimulation/{sub}/ses-*/{sub}_ses-*_run-*_neurovascularCoupling.log'))

    for log in logFiles:
        # print(f'processing {log}')
        logFile = pd.read_csv(log,usecols=[0])

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
        # logFile = pd.read_csv(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log', sep = '\t',skiprows=firstVolRow, names = ColNames)

        logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

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
            # print(f'comparing {target} with {targetDetected[i]}')
            if target == targetDetected[i]:
                count = count + 1
        if count >= 1:
            print(targetDetected)
            print(log)
            print(f'detected {count+1} missed targets in a row')

        # Count missed targets
        detectedRatio = (len(targetTimes) - targetDetected.count(-1))/len(targetTimes)

        subList.append(sub)
        runList.append(log.split('/')[-1][:20])
        targetDetectedList.append(detectedRatio)

data = pd.DataFrame({'subject': subList, 'run': runList, 'ratio': targetDetectedList})

ratios = []
counts = []

for ratio in data['ratio'].unique():
    tmp = data.loc[data['ratio'] == ratio]
    ratios.append(np.round(ratio,2))
    counts.append(len(tmp))

data2 = pd.DataFrame({'ratio': ratios, 'count': counts})

fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))
splot = sns.barplot(data=data2, x='ratio', y='count', linewidth=1, edgecolor="1", color='tab:red')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points',
                   fontsize=14)
plt.ylim(0, 90)
ax1.set_ylabel(r'#runs', fontsize=24)
ax1.set_xlabel(r'Ratio detected', fontsize=24)
ax1.yaxis.set_tick_params(labelsize=18)
ax1.xaxis.set_tick_params(labelsize=18)
fig.tight_layout()
plt.savefig(f'./results/targetsDetectedVsUndetected.png', bbox_inches="tight")

plt.show()


# sns.displot(data=data, x='ratio', bins=20)
# data.loc[data['ratio'] <= 0.78]
