'''

Generating event-files.tsv to be stored with data from log files

'''

import numpy as np
import glob
import pandas as pd
import os
import re


# define ROOT dir
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# define subjects to work on
subs = ['sub-03']

for sub in subs:
    # get all runs of all sessions
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-*_task-*_run-0*_*.nii.gz'))

    for run in runs[:]:
        if 'pilot' in run:
            runs.remove(run)

    for run in runs[::2]:
        # get basename of current run
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        # see session in which it was acquired
        if 'ses-01' in base:
            ses='ses-01'
        if 'ses-02' in base:
            ses='ses-02'


        if 'SingleShot' in base:
            logFile = pd.read_csv(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-01.log',usecols=[0])
        if 'MultiShot' in base:
            logFile = pd.read_csv(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-03.log',usecols=[0])

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
        if 'SingleShot' in base:
            logFile = pd.read_csv(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-01.log', sep = '\t',skiprows=firstVolRow, names = ColNames)
        if 'MultiShot' in base:
            logFile = pd.read_csv(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-03.log', sep = '\t',skiprows=firstVolRow, names = ColNames)
        # initiate lists
        stimStart = []
        stimStop = []


        # loop over lines and fine stimulation start and stop times
        for index, row in logFile.iterrows():


            if not logFile['event'][index] != logFile['event'][index]:
                if re.search('stimulation started', logFile['event'][index]):
                    stimStart.append(logFile['startTime'][index])
                if re.search('stimulation stopped', logFile['event'][index]):
                    stimStop.append(logFile['startTime'][index])

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)

        # make dataframe and save as text file
        design = pd.DataFrame({'onset': stimStart, 'duration': durs, 'trial_type' : ['stimulation']*len(durs)})
        design.to_csv(f'{ROOT}/{sub}/{ses}/func/{base}events.tsv',index=False)
