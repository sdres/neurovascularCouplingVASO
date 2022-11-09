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
subs = ['sub-07']

for sub in subs:
    # get all runs of all sessions
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-*_task-*_run-0*_*part-mag*_cbv.nii.gz'))

    for run in runs:
        # get basename of current run
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        # see session in which it was acquired

        for i in range(1,99):
            if f'ses-0{i}' in base:
                ses = f'ses-0{i}'

        log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log'
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
        stimStart = []
        stimStop = []
        stimDurs = []


        # loop over lines and fine stimulation start and stop times
        for index, row in logFile.iterrows():
            if not logFile['event'][index] != logFile['event'][index]:
                if re.search('stimulation started', logFile['event'][index]):
                    stimStart.append(logFile['startTime'][index])
                if re.search('stimulation stopped', logFile['event'][index]):
                    stimStop.append(logFile['startTime'][index])
                if re.search('stimDur', logFile['event'][index]):
                    stimDurs.append(int(float(re.findall(r"\d+\.\d+", logFile['event'][index])[0])))

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)
        stim = [f'stim {stimDur}s' for stimDur in stimDurs]

        # make dataframe and save as text file
        design = pd.DataFrame({'onset': stimStart, 'duration': durs, 'trial_type' : stim})
        for modality in ['bold', 'cbv']:
            design.to_csv(f'{ROOT}/{sub}/{ses}/func/{base}_{modality}_events.tsv',index=False)
