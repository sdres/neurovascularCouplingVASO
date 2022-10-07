'''

Creating stimulation times from log files and saving in bids format

'''

import numpy as np
import glob
import pandas as pd
import os
import re

import seaborn as sns
import matplotlib.pyplot as plt

# Define ROOT dir
ROOT = '/Users/sebastiandresbach/git/neurovascularCouplingVASO'

# Define subjects to work on
SUBS = ['sub-01']


for sub in SUBS:
    # get all runs of all sessions
    logFiles = sorted(glob.glob(f'{ROOT}/code/stimulation/{sub}/ses-*/{sub}_ses-*_run-0*_neurovascularCoupling.log'))

    for logFile in logFiles:
        # get basename of current run
        base = os.path.basename(logFile).rsplit('.', 2)[0]
        # get logfile
        data = pd.read_csv(f'{logFile}'
                                ,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in data.iterrows():
            if re.search('StartOfRun', str(row)):
                firstVolRow = index+1
                break
        # define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        data = pd.read_csv(f'{logFile}', sep = '\t', skiprows=firstVolRow, names = ColNames)

        # initiate lists
        stimStart = []
        stimStop = []

        # loop over lines and fine stimulation start and stop times
        for index, row in data.iterrows():
            if not data['event'][index] != data['event'][index]:
                if re.search('stimulation started', data['event'][index]):
                    stimStart.append(data['startTime'][index])
                if re.search('stimulation stopped', data['event'][index]):
                    stimStop.append(data['startTime'][index])

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)

        trials = ['stimulation']*len(durs)

        # make dataframe and save as text file
        design = pd.DataFrame({'onset': stimStart, 'duration': durs, 'trial_type' : trials})

        design.to_csv(f'{ROOT}/{sub}/{ses}/{base}_events.tsv',
                      sep = ' ',
                      index = False
                      )
