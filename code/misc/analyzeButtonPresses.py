'''

Investigate attention task performance.

'''

import numpy as np
import glob
import pandas as pd
import os
import re


# define ROOT dir
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# define subjects to work on
subs = ['sub-05','sub-06','sub-07','sub-08']

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
